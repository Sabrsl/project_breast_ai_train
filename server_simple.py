#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BreastAI WebSocket Server v3.5 - FIXED
Corrections: Unicode safe (Windows), broadcast vraiment fonctionnel
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set
from enum import Enum

import websockets
from websockets.server import WebSocketServerProtocol

# FIX UNICODE pour Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Importer le système d'entraînement
from breastai_training import ClinicalTrainingSystem, Config

# Configuration logging SANS émojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/server.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==================================================================================
# ENUMS ET CONSTANTES
# ==================================================================================

class ServerState(Enum):
    """États du serveur"""
    IDLE = "idle"
    TRAINING = "training"
    STOPPING = "stopping"
    EXPORTING = "exporting"
    ERROR = "error"

# Constantes
MAX_QUEUE_SIZE = 2000
BROADCAST_TIMEOUT = 3.0
WORKER_CHECK_INTERVAL = 0.5
MAX_CHECKPOINT_SIZE_MB = 500
PING_INTERVAL = 30

# ==================================================================================
# SERVEUR WEBSOCKET
# ==================================================================================

class BreastAIServer:
    """Serveur WebSocket pour BreastAI avec affichage temps réel garanti"""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.training_system: Optional[ClinicalTrainingSystem] = None
        self.training_task: Optional[asyncio.Task] = None
        
        # État du serveur
        self.state = ServerState.IDLE
        
        # Queue de messages avec limite
        self.message_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.broadcast_task = None
        self._shutdown_event = asyncio.Event()
        
        # Compteurs statistiques
        self.stats = {
            'messages_sent': 0,
            'messages_dropped': 0,
            'clients_total': 0,
            'errors': 0
        }
        
        logger.info(f"[OK] Serveur initialise sur {host}:{port}")
    
    @property
    def is_training(self) -> bool:
        """État training"""
        return self.state == ServerState.TRAINING
    
    async def broadcast(self, message: Dict):
        """
        Broadcast 100% non-bloquant
        CRITIQUE: Cette fonction est appelée depuis le training thread
        """
        try:
            # Ajouter timestamp si absent
            if 'timestamp' not in message:
                message['timestamp'] = datetime.now().isoformat()
            
            # put_nowait = vraiment non-bloquant (pas d'await)
            self.message_queue.put_nowait(message)
            
            # DEBUG: Print pour vérifier que les messages passent
            msg_type = message.get('type', 'unknown')
            
        except asyncio.QueueFull:
            self.stats['messages_dropped'] += 1
            if self.stats['messages_dropped'] % 100 == 0:
                logger.warning(f"Queue saturee: {self.stats['messages_dropped']} messages perdus")
        except Exception as e:
            logger.error(f"Erreur broadcast: {e}")
            self.stats['errors'] += 1
    
    async def broadcast_worker(self):
        """Worker qui envoie les messages"""
        logger.info("[WORKER] Broadcast worker demarre")
        
        while not self._shutdown_event.is_set():
            try:
                # Attendre message avec timeout
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=WORKER_CHECK_INTERVAL
                    )
                except asyncio.TimeoutError:
                    continue
                
                if not self.clients:
                    logger.warning("[WORKER] Aucun client connecte, message ignore")
                    continue
                
                # Sérialiser une seule fois
                try:
                    message_json = json.dumps(message)
                except (TypeError, ValueError) as e:
                    logger.error(f"Erreur JSON: {e}")
                    continue
                                
                # Copie pour itération sûre
                clients_copy = self.clients.copy()
                disconnected = set()
                
                # Envoi à tous les clients
                for client in clients_copy:
                    try:
                        await asyncio.wait_for(
                            client.send(message_json),
                            timeout=BROADCAST_TIMEOUT
                        )
                        self.stats['messages_sent'] += 1
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout envoi client {client.remote_address}")
                        disconnected.add(client)
                        
                    except websockets.exceptions.ConnectionClosed:
                        logger.info(f"Client deconnecte: {client.remote_address}")
                        disconnected.add(client)
                        
                    except Exception as e:
                        logger.warning(f"Erreur envoi: {e}")
                        disconnected.add(client)
                
                # Cleanup clients déconnectés
                if disconnected:
                    self.clients -= disconnected
                    logger.info(f"Clients retires: {len(disconnected)}, reste: {len(self.clients)}")
                
            except Exception as e:
                logger.error(f"Erreur broadcast_worker: {e}", exc_info=True)
                self.stats['errors'] += 1
                await asyncio.sleep(0.1)
        
        logger.info("[WORKER] Broadcast worker arrete")
    
    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Gère une connexion client"""
        self.clients.add(websocket)
        self.stats['clients_total'] += 1
        client_addr = websocket.remote_address
        logger.info(f"[CLIENT] Connecte: {client_addr} (Total: {len(self.clients)})")
        
        # Tâche keep-alive
        keep_alive_task = None
        
        try:
            # Message de bienvenue IMMÉDIAT
            welcome_msg = {
                'type': 'connection_established',
                'message': 'Connecte au serveur BreastAI v3.5',
                'server_state': self.state.value,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Envoyer état initial si training en cours
            if self.is_training and self.training_system:
                await websocket.send(json.dumps({
                    'type': 'log',
                    'message': f'Entrainement en cours (epoch {getattr(self.training_system, "current_epoch", 0)})',
                    'level': 'warning'
                }))
            
            # Keep-alive
            async def keep_alive():
                try:
                    while websocket in self.clients and not websocket.closed:
                        await asyncio.sleep(PING_INTERVAL)
                        if websocket in self.clients:
                            await websocket.ping()
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning(f"Keep-alive error: {e}")
            
            keep_alive_task = asyncio.create_task(keep_alive())
            
            # Boucle de réception
            async for message in websocket:
                await self.handle_message(websocket, message)
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[CLIENT] Connexion fermee: {client_addr}")
        except Exception as e:
            logger.error(f"[CLIENT] Erreur {client_addr}: {e}", exc_info=True)
            try:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': f'Erreur serveur: {str(e)}',
                    'fatal': True
                }))
            except:
                pass
        finally:
            self.clients.discard(websocket)
            if keep_alive_task and not keep_alive_task.done():
                keep_alive_task.cancel()
            logger.info(f"[CLIENT] Retire: {client_addr} (Restants: {len(self.clients)})")
    
    async def handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Traite un message reçu"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            logger.info(f"[MESSAGE] Recu: {msg_type}")
            
            # Router les messages
            if msg_type == 'start_training':
                config = data.get('config', data)
                await self.start_training(config)
            
            elif msg_type == 'stop_training':
                await self.stop_training()
            
            elif msg_type == 'get_status':
                await self.send_status(websocket)
            
            elif msg_type == 'list_checkpoints':
                await self.list_checkpoints()
            
            elif msg_type == 'load_checkpoint':
                checkpoint = data.get('checkpoint')
                await self.load_checkpoint(checkpoint)
            
            elif msg_type == 'resume_training':
                checkpoint = data.get('checkpoint')
                epochs = data.get('epochs')
                await self.resume_training(checkpoint, epochs)
            
            elif msg_type == 'delete_checkpoint':
                checkpoint = data.get('checkpoint')
                await self.delete_checkpoint(checkpoint)
            
            elif msg_type == 'export_onnx':
                await self.export_onnx(data)
            
            elif msg_type == 'system_diagnostics':
                await self.system_diagnostics()
            
            elif msg_type == 'ping':
                await websocket.send(json.dumps({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }))
            
            elif msg_type == 'disconnect':
                logger.info(f"[MESSAGE] Deconnexion demandee: {websocket.remote_address}")
                await websocket.send(json.dumps({
                    'type': 'disconnect_ack',
                    'message': 'Au revoir!',
                    'timestamp': datetime.now().isoformat()
                }))
                await websocket.close()
            
            else:
                logger.warning(f"[MESSAGE] Type inconnu: {msg_type}")
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': f'Type de message non supporte: {msg_type}'
                }))
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON invalide: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Format JSON invalide'
            }))
        except Exception as e:
            logger.error(f"Erreur traitement message: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur serveur: {str(e)}'
            })
    
    async def start_training(self, config_data: Dict):
        """Démarre un nouvel entraînement"""
        if self.is_training:
            await self.broadcast({
                'type': 'error',
                'message': 'Entrainement deja en cours'
            })
            return
        
        try:
            self.state = ServerState.TRAINING
            
            await self.broadcast({
                'type': 'log',
                'message': 'Initialisation de l\'entrainement...',
                'level': 'info'
            })
            
            # Mapper la config
            mapped_config = self._map_interface_config(config_data)
            
            await self.broadcast({
                'type': 'log',
                'message': f'Configuration: {mapped_config["model"]["architecture"]}, {mapped_config["training"]["epochs"]} epochs',
                'level': 'info'
            })
            
            # Créer le système avec callback
            config = Config(mapped_config)
            self.training_system = ClinicalTrainingSystem(config, callback=self.broadcast)
            
            await self.broadcast({
                'type': 'log',
                'message': 'Chargement du modele et des donnees...',
                'level': 'info'
            })
            
            # Setup
            success = await self.training_system.setup()
            if not success:
                raise ValueError("Echec de l'initialisation du systeme")
            
            await self.broadcast({
                'type': 'training_started',
                'message': 'Entrainement demarre!',
                'config': mapped_config
            })
            
            # Lancer l'entraînement
            epochs = config_data.get('epochs', 50)
            self.training_task = asyncio.create_task(self._run_training(epochs))
            
            logger.info(f"[TRAINING] Lance: {epochs} epochs")
            
        except Exception as e:
            logger.error(f"Erreur start_training: {e}", exc_info=True)
            self.state = ServerState.ERROR
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur demarrage: {str(e)}'
            })
            await self._cleanup_training_system()
    
    async def _run_training(self, epochs: int):
        """Execute l'entraînement"""
        try:
            await self.training_system.train(epochs)
            
            await self.broadcast({
                'type': 'training_complete',
                'message': 'Entrainement termine avec succes!'
            })
            
        except asyncio.CancelledError:
            logger.info("Entrainement annule")
            await self.broadcast({
                'type': 'training_stopped',
                'message': 'Entrainement arrete par l\'utilisateur'
            })
            raise
            
        except Exception as e:
            logger.error(f"Erreur entrainement: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur entrainement: {str(e)}'
            })
            
        finally:
            self.state = ServerState.IDLE
            await self._cleanup_training_system()
    
    async def _cleanup_training_system(self):
        """Cleanup complet"""
        if self.training_system:
            try:
                logger.info("Nettoyage du systeme d'entrainement...")
                
                if hasattr(self.training_system, 'cleanup'):
                    await self.training_system.cleanup()
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cache GPU vide")
                except:
                    pass
                
                self.training_system = None
                logger.info("Cleanup termine")
                
            except Exception as e:
                logger.error(f"Erreur cleanup: {e}", exc_info=True)
    
    async def stop_training(self):
        """Arrête l'entraînement"""
        if not self.is_training or self.training_system is None:
            await self.broadcast({
                'type': 'error',
                'message': 'Aucun entrainement en cours'
            })
            return
        
        logger.info("Arret de l'entrainement demande")
        self.state = ServerState.STOPPING
        
        await self.broadcast({
            'type': 'log',
            'message': 'Arret en cours (sauvegarde checkpoint)...',
            'level': 'warning'
        })
        
        await self.training_system.stop()
        
        if self.training_task and not self.training_task.done():
            self.training_task.cancel()
            try:
                await self.training_task
            except asyncio.CancelledError:
                pass
    
    async def send_status(self, websocket: WebSocketServerProtocol):
        """Envoie le statut"""
        status = {
            'type': 'status',
            'state': self.state.value,
            'is_training': self.is_training,
            'clients_connected': len(self.clients),
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.training_system:
            status['training_info'] = {
                'current_epoch': getattr(self.training_system, 'current_epoch', 0),
                'best_val_acc': getattr(self.training_system, 'best_val_acc', 0.0)
            }
        
        await websocket.send(json.dumps(status))
    
    def _normalize_checkpoint_path(self, checkpoint_input: str) -> Path:
        """Normalise les chemins"""
        checkpoint_dir = Path('checkpoints')
        checkpoint_str = str(checkpoint_input).strip()
        
        if checkpoint_str.startswith('checkpoints') and not checkpoint_str.startswith(('checkpoints/', 'checkpoints\\')):
            filename = checkpoint_str[len('checkpoints'):]
            full_path = checkpoint_dir / filename
            return full_path
        
        path = Path(checkpoint_str)
        
        if 'checkpoints' in path.parts:
            return path
        
        full_path = checkpoint_dir / path.name
        return full_path
    
    def _validate_checkpoint_path(self, path: Path) -> bool:
        """Valide les chemins"""
        try:
            checkpoint_dir = Path('checkpoints').resolve()
            resolved_path = path.resolve()
            return resolved_path.is_relative_to(checkpoint_dir)
        except:
            return False
    
    async def list_checkpoints(self):
        """Liste les checkpoints"""
        try:
            import torch
            
            checkpoint_dir = Path('checkpoints')
            if not checkpoint_dir.exists():
                await self.broadcast({
                    'type': 'checkpoints_list',
                    'checkpoints': []
                })
                return
            
            await self.broadcast({
                'type': 'log',
                'message': 'Chargement des checkpoints...',
                'level': 'info'
            })
            
            checkpoints = []
            for ckpt_file in checkpoint_dir.glob('*.pth'):
                try:
                    stat = ckpt_file.stat()
                    size_mb = stat.st_size / (1024 * 1024)
                    
                    if size_mb > MAX_CHECKPOINT_SIZE_MB:
                        continue
                    
                    try:
                        import numpy
                        import torch.serialization
                        
                        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
                            ckpt_data = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                        
                        epoch = ckpt_data.get('epoch', 0)
                        accuracy = ckpt_data.get('best_val_acc', 0.0)
                        architecture = ckpt_data.get('architecture', 'unknown')
                        timestamp = ckpt_data.get('timestamp', '')
                    except Exception as e:
                        logger.warning(f"Erreur lecture metadonnees {ckpt_file.name}: {e}")
                        epoch = 0
                        accuracy = 0.0
                        architecture = 'unknown'
                        timestamp = ''
                    
                    checkpoints.append({
                        'filename': ckpt_file.name,
                        'path': str(ckpt_file),
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'epoch': epoch,
                        'accuracy': float(accuracy),
                        'architecture': architecture,
                        'type': 'best' if 'best' in ckpt_file.name else 'periodic',
                        'timestamp': timestamp
                    })
                except Exception as e:
                    logger.warning(f"Erreur lecture checkpoint {ckpt_file}: {e}")
            
            checkpoints.sort(key=lambda x: x['created'], reverse=True)
            
            await self.broadcast({
                'type': 'checkpoints_list',
                'checkpoints': checkpoints
            })
            
            logger.info(f"Liste de {len(checkpoints)} checkpoints envoyee")
            
        except Exception as e:
            logger.error(f"Erreur list_checkpoints: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur liste checkpoints: {str(e)}'
            })
    
    async def load_checkpoint(self, checkpoint_path: str):
        """Charge un checkpoint"""
        try:
            await self._cleanup_training_system()
            
            normalized_path = self._normalize_checkpoint_path(checkpoint_path)
            
            if not self._validate_checkpoint_path(normalized_path):
                raise ValueError(f"Chemin non autorise: {checkpoint_path}")
            
            if not normalized_path.exists():
                raise FileNotFoundError(f"Checkpoint introuvable: {normalized_path}")
            
            await self.broadcast({
                'type': 'log',
                'message': f'Chargement checkpoint: {normalized_path.name}',
                'level': 'info'
            })
            
            import torch
            import numpy
            import torch.serialization
            
            with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
                checkpoint_data = torch.load(normalized_path, map_location='cpu', weights_only=False)
            
            saved_config = checkpoint_data.get('config', {})
            
            await self.broadcast({
                'type': 'log',
                'message': f'Checkpoint epoch {checkpoint_data.get("epoch", 0)}, accuracy {checkpoint_data.get("best_val_acc", 0):.2f}%',
                'level': 'info'
            })
            
            config = Config(saved_config)
            self.training_system = ClinicalTrainingSystem(config, callback=self.broadcast)
            
            await self.broadcast({
                'type': 'log',
                'message': 'Initialisation du systeme...',
                'level': 'info'
            })
            
            success = await self.training_system.setup()
            if not success:
                raise ValueError("Echec du setup")
            
            start_epoch = self.training_system.load_checkpoint(str(normalized_path))
            if start_epoch is None:
                raise ValueError("Echec du chargement du checkpoint")
            
            await self.broadcast({
                'type': 'checkpoint_loaded',
                'checkpoint': str(normalized_path),
                'start_epoch': start_epoch + 1,
                'best_val_acc': self.training_system.best_val_acc,
                'message': f'Pret a reprendre depuis epoch {start_epoch + 1}'
            })
            
            logger.info(f"Checkpoint charge: epoch {start_epoch}")
            
        except FileNotFoundError as e:
            logger.error(f"{e}")
            await self.broadcast({
                'type': 'error',
                'message': f'Checkpoint introuvable: {checkpoint_path}'
            })
            await self._cleanup_training_system()
        except Exception as e:
            logger.error(f"Erreur load_checkpoint: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur chargement: {str(e)}'
            })
            await self._cleanup_training_system()
    
    async def resume_training(self, checkpoint_path: str, epochs: Optional[int] = None):
        """Reprend l'entraînement"""
        try:
            import torch
            
            if self.is_training:
                await self.broadcast({
                    'type': 'error',
                    'message': 'Entrainement deja en cours'
                })
                return
            
            normalized_path = self._normalize_checkpoint_path(checkpoint_path)
            
            if not self._validate_checkpoint_path(normalized_path):
                raise ValueError(f"Chemin non autorise: {checkpoint_path}")
            
            if not normalized_path.exists():
                raise FileNotFoundError(f"Checkpoint introuvable: {normalized_path}")
            
            await self.load_checkpoint(str(normalized_path))
            
            if not self.training_system:
                raise ValueError("Systeme d'entrainement non initialise")
            
            import numpy
            import torch.serialization
            
            with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
                checkpoint_data = torch.load(normalized_path, map_location='cpu', weights_only=False)
            
            start_epoch = checkpoint_data.get('epoch', 0) + 1
            
            if epochs is None:
                epochs = self.training_system.config.get('training', 'epochs', default=50)
            
            await self.broadcast({
                'type': 'log',
                'message': f'Reprise entrainement epoch {start_epoch} -> {epochs}',
                'level': 'info'
            })
            
            self.state = ServerState.TRAINING
            self.training_task = asyncio.create_task(
                self.training_system.train(epochs=epochs, start_epoch=start_epoch)
            )
            
            logger.info(f"Entrainement repris depuis epoch {start_epoch}")
            
        except FileNotFoundError as e:
            logger.error(f"{e}")
            await self.broadcast({
                'type': 'error',
                'message': f'Checkpoint introuvable: {checkpoint_path}'
            })
        except Exception as e:
            logger.error(f"Erreur resume_training: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur reprise: {str(e)}'
            })
            self.state = ServerState.IDLE
    
    async def delete_checkpoint(self, checkpoint: str):
        """Supprime un checkpoint"""
        try:
            ckpt_path = self._normalize_checkpoint_path(checkpoint)
            
            if not self._validate_checkpoint_path(ckpt_path):
                raise ValueError(f"Chemin non autorise: {checkpoint}")
            
            if ckpt_path.exists():
                ckpt_path.unlink()
                await self.broadcast({
                    'type': 'log',
                    'message': f'Checkpoint supprime: {ckpt_path.name}',
                    'level': 'success'
                })
                await self.list_checkpoints()
            else:
                await self.broadcast({
                    'type': 'error',
                    'message': f'Checkpoint introuvable: {ckpt_path}'
                })
        except Exception as e:
            logger.error(f"Erreur delete_checkpoint: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur suppression: {str(e)}'
            })
    
    async def export_onnx(self, data: Dict):
        """Exporte en ONNX"""
        temp_system = None
        try:
            if self.is_training:
                await self.broadcast({
                    'type': 'error',
                    'message': 'Export impossible pendant entrainement'
                })
                return
            
            checkpoint_path = data.get('checkpoint')
            normalized_path = self._normalize_checkpoint_path(checkpoint_path)
            
            if not self._validate_checkpoint_path(normalized_path):
                raise ValueError(f"Chemin non autorise: {checkpoint_path}")
            
            if not normalized_path.exists():
                raise FileNotFoundError(f"Checkpoint introuvable: {normalized_path}")
            
            await self.broadcast({
                'type': 'log',
                'message': f'Debut export ONNX: {normalized_path.name}',
                'level': 'info'
            })
            
            if not self.training_system:
                await self.broadcast({
                    'type': 'log',
                    'message': 'Initialisation temporaire pour export...',
                    'level': 'info'
                })
                
                import torch
                import numpy
                import torch.serialization
                
                with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
                    checkpoint_data = torch.load(normalized_path, map_location='cpu', weights_only=False)
                
                if 'config' in checkpoint_data:
                    saved_config = checkpoint_data['config']
                    config = Config(saved_config)
                else:
                    architecture = checkpoint_data.get('architecture', 'efficientnetv2_s')
                    num_classes = checkpoint_data.get('num_classes', 3)
                    use_cbam = checkpoint_data.get('use_cbam', True)
                    image_size = checkpoint_data.get('image_size', 512)
                    
                    config_dict = {
                        'model': {
                            'architecture': architecture,
                            'num_classes': num_classes,
                            'use_cbam': use_cbam,
                            'dropout_rate': 0.4
                        },
                        'data': {
                            'image_size': image_size,
                            'batch_size': 1
                        },
                        'paths': {
                            'export_dir': 'exports',
                            'checkpoint_dir': 'checkpoints'
                        }
                    }
                    config = Config(config_dict)
                
                temp_system = ClinicalTrainingSystem(config, callback=self.broadcast)
                await temp_system.setup()
                success = await temp_system.export_onnx(str(normalized_path))
                
                del temp_system
                temp_system = None
            else:
                success = await self.training_system.export_onnx(str(normalized_path))
            
            if success:
                logger.info("Export ONNX reussi")
            else:
                logger.error("Export ONNX echoue")
                
        except FileNotFoundError as e:
            logger.error(f"{e}")
            await self.broadcast({
                'type': 'error',
                'message': f'Checkpoint introuvable: {checkpoint_path}'
            })
        except Exception as e:
            logger.error(f"Erreur export_onnx: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur export: {str(e)}'
            })
        finally:
            if temp_system:
                try:
                    del temp_system
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
    
    async def system_diagnostics(self):
        """Diagnostics système"""
        try:
            import psutil
            
            await self.broadcast({
                'type': 'log',
                'message': 'Collecte des diagnostics systeme...',
                'level': 'info'
            })
            
            diagnostics = {
                'type': 'system_diagnostics',
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_percent': psutil.disk_usage('/').percent,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.broadcast(diagnostics)
            logger.info("Diagnostics systeme envoyes")
            
        except Exception as e:
            logger.error(f"Erreur system_diagnostics: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur diagnostics: {str(e)}'
            })
    
    def _map_interface_config(self, interface_config: Dict) -> Dict:
        """Mappe la config interface vers structure interne"""
        config = {
            'paths': {
                'data_dir': interface_config.get('data_path', 'data'),
                'checkpoint_dir': interface_config.get('checkpoint_path', 'checkpoints')
            },
            'training': {
                'epochs': interface_config.get('epochs', 50),
                'learning_rate': interface_config.get('learning_rate', 0.0003),
                'weight_decay': interface_config.get('weight_decay', 0.0001),
                'label_smoothing': interface_config.get('label_smoothing', 0.1),
                'use_ema': interface_config.get('use_ema', False),
                'ema_decay': interface_config.get('ema_decay', 0.9998),
                'focal_loss': {
                    'enabled': interface_config.get('use_focal_loss', False),
                    'alpha': interface_config.get('focal_loss_alpha', [0.25, 0.50, 0.25]),
                    'gamma': interface_config.get('focal_loss_gamma', 2.5)
                }
            },
            'data': {
                'batch_size': interface_config.get('batch_size', 4),
                'num_workers': interface_config.get('num_workers', 4),
                'image_size': interface_config.get('image_size', 512),
                'gradient_accumulation_steps': interface_config.get('gradient_accumulation_steps', 1)
            },
            'inference': {
                'tta_enabled': interface_config.get('use_tta', False)
            },
            'model': {
                'architecture': interface_config.get('model_name', 'efficientnetv2_s'),
                'num_classes': interface_config.get('num_classes', 3),
                'use_cbam': interface_config.get('use_cbam', True),
                'dropout_rate': interface_config.get('dropout_rate', 0.4),
                'progressive_unfreezing': {
                    'phase1_epochs': interface_config.get('progressive_unfreezing_phase1', 8),
                    'phase2_epochs': interface_config.get('progressive_unfreezing_phase2', 20),
                    'phase3_epochs': interface_config.get('progressive_unfreezing_phase3', 40)
                }
            }
        }
        
        logger.info(f"Config mappee: model={config['model']['architecture']}, "
                   f"epochs={config['training']['epochs']}, "
                   f"batch={config['data']['batch_size']}")
        
        return config
    
    async def shutdown(self):
        """Arrêt propre du serveur"""
        logger.info("Arret du serveur...")
        
        if self.is_training:
            await self.stop_training()
            if self.training_task:
                try:
                    await asyncio.wait_for(self.training_task, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout arret training")
        
        self._shutdown_event.set()
        
        if self.broadcast_task and not self.broadcast_task.done():
            try:
                await asyncio.wait_for(self.broadcast_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout arret broadcast worker")
                self.broadcast_task.cancel()
        
        await self._cleanup_training_system()
        
        if self.clients:
            logger.info(f"Fermeture de {len(self.clients)} client(s)...")
            for client in self.clients.copy():
                try:
                    await client.close()
                except:
                    pass
        
        logger.info("Serveur arrete proprement")
        logger.info(f"Stats finales: {self.stats}")
    
    async def start(self):
        """Démarre le serveur"""
        logger.info("="*80)
        logger.info("BREASTAI WEBSOCKET SERVER v3.5 - FIXED")
        logger.info(f"Demarrage sur ws://{self.host}:{self.port}")
        logger.info("="*80)
        
        # Démarrer le broadcast worker
        self.broadcast_task = asyncio.create_task(self.broadcast_worker())
        logger.info("Broadcast worker demarre")
        
        try:
            async with websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=PING_INTERVAL,
                ping_timeout=PING_INTERVAL * 2
            ):
                logger.info("Serveur pret! En attente de connexions...")
                logger.info(f"Queue size: {MAX_QUEUE_SIZE}, Broadcast timeout: {BROADCAST_TIMEOUT}s")
                await asyncio.Future()
        finally:
            await self.shutdown()

# ==================================================================================
# POINT D'ENTREE
# ==================================================================================

async def main():
    """Point d'entree principal"""
    Path('logs').mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    Path('exports').mkdir(exist_ok=True)
    
    logger.info("Dossiers initialises")
    
    server = BreastAIServer(host='localhost', port=8765)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("\nArret demande par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
    finally:
        if hasattr(server, 'shutdown'):
            try:
                await server.shutdown()
            except:
                pass

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nAu revoir!")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
        sys.exit(1)