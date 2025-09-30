#!/usr/bin/env python3
"""
BreastAI WebSocket Server v3.3.1 - PRODUCTION READY
Serveur WebSocket pour controle de l'entrainement depuis l'interface web
Corrections: chemins checkpoints, gestion erreurs, code production
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

# Importer le systeme d'entrainement
from breastai_training import TrainingSystem, Config

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/server.log')
    ]
)
logger = logging.getLogger(__name__)

# ==================================================================================
# SERVEUR WEBSOCKET
# ==================================================================================

class BreastAIServer:
    """Serveur WebSocket pour BreastAI"""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.training_system: Optional[TrainingSystem] = None
        self.training_task: Optional[asyncio.Task] = None
        self.is_training = False
        
        logger.info(f"Serveur initialise sur {host}:{port}")
    
    async def broadcast(self, message: Dict):
        """Broadcast un message a tous les clients connectes"""
        if not self.clients:
            return
        
        try:
            message_json = json.dumps(message)
            clients_copy = self.clients.copy()
            disconnected = set()
            
            for client in clients_copy:
                try:
                    await client.send(message_json)
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"Client deconnecte pendant broadcast: {client.remote_address}")
                    disconnected.add(client)
                except Exception as e:
                    logger.warning(f"Erreur envoi a client {client.remote_address}: {e}")
                    disconnected.add(client)
            
            if disconnected:
                self.clients -= disconnected
                logger.info(f"Clients deconnectes: {len(disconnected)} (Restants: {len(self.clients)})")
                
        except Exception as e:
            logger.error(f"Erreur broadcast: {e}")
    
    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Gere une connexion client"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Client connecte: {client_addr}")
        
        try:
            # Message de bienvenue
            await websocket.send(json.dumps({
                'type': 'connection_established',
                'message': 'Connecte au serveur BreastAI',
                'timestamp': datetime.now().isoformat()
            }))
            
            # Keep-alive
            async def keep_alive():
                try:
                    while websocket in self.clients:
                        await asyncio.sleep(30)
                        if websocket in self.clients:
                            await websocket.ping()
                except:
                    pass
            
            asyncio.create_task(keep_alive())
            
            # Boucle de reception
            async for message in websocket:
                await self.handle_message(websocket, message)
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client deconnecte: {client_addr}")
            logger.info(f"Clients restants: {len(self.clients) - 1}")
        except Exception as e:
            logger.error(f"Erreur client {client_addr}: {e}", exc_info=True)
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client retire: {client_addr} (Clients restants: {len(self.clients)})")
    
    async def handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Traite un message recu"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            logger.info(f"Message recu: {msg_type}")
            
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
                await websocket.send(json.dumps({'type': 'pong'}))
            
            elif msg_type == 'disconnect':
                logger.info(f"Client demande deconnexion: {websocket.remote_address}")
                await websocket.send(json.dumps({
                    'type': 'disconnect_ack',
                    'message': 'Deconnexion confirmee',
                    'timestamp': datetime.now().isoformat()
                }))
                await websocket.close()
                return
            
            else:
                logger.warning(f"Type de message inconnu: {msg_type}")
        
        except json.JSONDecodeError:
            logger.error("Message JSON invalide")
        except Exception as e:
            logger.error(f"Erreur traitement message: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur serveur: {str(e)}'
            })
    
    async def start_training(self, config_data: Dict):
        """Demarre un nouvel entrainement"""
        if self.is_training:
            await self.broadcast({
                'type': 'error',
                'message': 'Entrainement deja en cours'
            })
            return
        
        try:
            logger.info("Demarrage entrainement")
            
            # Mapper la config
            mapped_config = self._map_interface_config(config_data)
            
            # Creer le systeme
            config = Config(mapped_config)
            self.training_system = TrainingSystem(config, callback=self.broadcast)
            
            # Setup
            success = await self.training_system.setup()
            if not success:
                await self.broadcast({
                    'type': 'error',
                    'message': 'Echec de l\'initialisation'
                })
                return
            
            # Lancer l'entrainement
            self.is_training = True
            epochs = config_data.get('epochs', 50)
            self.training_task = asyncio.create_task(self._run_training(epochs))
            
            logger.info(f"Entrainement lance: {epochs} epochs")
            
        except Exception as e:
            logger.error(f"Erreur start_training: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur demarrage: {str(e)}'
            })
            self.is_training = False
    
    async def _run_training(self, epochs: int):
        """Execute l'entrainement"""
        try:
            await self.training_system.train(epochs)
        except Exception as e:
            logger.error(f"Erreur entrainement: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur entrainement: {str(e)}'
            })
        finally:
            self.is_training = False
            self.training_system = None
    
    async def stop_training(self):
        """Arrete l'entrainement en cours"""
        if not self.is_training or self.training_system is None:
            await self.broadcast({
                'type': 'error',
                'message': 'Aucun entrainement en cours'
            })
            return
        
        logger.info("Arret de l'entrainement demande")
        
        await self.training_system.stop()
        
        if self.training_task and not self.training_task.done():
            self.training_task.cancel()
            try:
                await self.training_task
            except asyncio.CancelledError:
                logger.info("Training task cancelled")
    
    async def send_status(self, websocket: WebSocketServerProtocol):
        """Envoie le statut actuel"""
        status = {
            'type': 'status',
            'is_training': self.is_training,
            'clients_connected': len(self.clients),
            'timestamp': datetime.now().isoformat()
        }
        await websocket.send(json.dumps(status))
    
    def _normalize_checkpoint_path(self, checkpoint_input: str) -> Path:
        """
        CORRECTION CRITIQUE: Normalise et repare les chemins malformes
        
        Probleme: L'interface peut envoyer des chemins malformes comme:
        - 'checkpointslatest_epoch_002.pth' (manque le separateur)
        - 'latest_epoch_002.pth' (manque le dossier)
        - 'checkpoints/latest_epoch_002.pth' (correct)
        - 'checkpoints\\latest_epoch_002.pth' (Windows)
        
        Args:
            checkpoint_input: Chemin potentiellement malformé
            
        Returns:
            Path valide: checkpoints/filename.pth
        """
        checkpoint_dir = Path('checkpoints')
        
        # 1. Nettoyer le chemin d'entree
        checkpoint_str = str(checkpoint_input).strip()
        
        # 2. CORRECTION CRITIQUE: Detecter et reparer 'checkpointsXXX.pth'
        if checkpoint_str.startswith('checkpoints') and not checkpoint_str.startswith(('checkpoints/', 'checkpoints\\')):
            # Malformé: 'checkpointslatest_epoch_002.pth'
            # Extraire le nom de fichier en retirant 'checkpoints'
            filename = checkpoint_str[len('checkpoints'):]
            logger.warning(f"Chemin malformé détecté: '{checkpoint_str}' -> extraction: '{filename}'")
            full_path = checkpoint_dir / filename
            logger.info(f"Réparation: '{checkpoint_str}' -> '{full_path}'")
            return full_path
        
        # 3. Convertir en Path pour manipulation
        path = Path(checkpoint_str)
        
        # 4. Si le chemin contient deja le dossier checkpoints (bien formé)
        if 'checkpoints' in path.parts:
            logger.debug(f"Chemin déjà complet: '{checkpoint_str}'")
            return path
        
        # 5. Sinon, c'est juste le nom de fichier
        full_path = checkpoint_dir / path.name
        logger.debug(f"Ajout du dossier: '{checkpoint_str}' -> '{full_path}'")
        
        return full_path
    
    async def list_checkpoints(self):
        """Liste les checkpoints disponibles avec metadonnees"""
        try:
            import torch
            
            checkpoint_dir = Path('checkpoints')
            if not checkpoint_dir.exists():
                await self.broadcast({
                    'type': 'checkpoints_list',
                    'checkpoints': []
                })
                return
            
            checkpoints = []
            for ckpt_file in checkpoint_dir.glob('*.pth'):
                try:
                    stat = ckpt_file.stat()
                    
                    # Charger les metadonnees avec PyTorch 2.6+ compatibility
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
            
            # Trier par date (plus recent d'abord)
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
        """Charge un checkpoint et initialise pour reprendre l'entrainement"""
        try:
            # CORRECTION: Normaliser le chemin
            normalized_path = self._normalize_checkpoint_path(checkpoint_path)
            
            # Verifier l'existence
            if not normalized_path.exists():
                raise FileNotFoundError(f"Checkpoint introuvable: {normalized_path}")
            
            await self.broadcast({
                'type': 'log',
                'message': f'Chargement checkpoint: {normalized_path.name}',
                'level': 'info'
            })
            
            # Charger les metadonnees avec PyTorch 2.6+ compatibility
            import torch
            import numpy
            import torch.serialization
            
            with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
                checkpoint_data = torch.load(normalized_path, map_location='cpu', weights_only=False)
            
            # Recuperer la config du checkpoint
            saved_config = checkpoint_data.get('config', {})
            
            await self.broadcast({
                'type': 'log',
                'message': f'Checkpoint epoch {checkpoint_data.get("epoch", 0)}, accuracy {checkpoint_data.get("best_val_acc", 0):.2f}%',
                'level': 'info'
            })
            
            # Creer le systeme d'entrainement avec la config sauvegardee
            from breastai_training import Config, TrainingSystem
            config = Config(saved_config)
            self.training_system = TrainingSystem(config, callback=self.broadcast)
            
            # Setup
            await self.broadcast({
                'type': 'log',
                'message': 'Initialisation du systeme...',
                'level': 'info'
            })
            
            success = await self.training_system.setup()
            
            if not success:
                raise ValueError("Echec du setup")
            
            # Charger le checkpoint (utiliser le chemin normalise)
            start_epoch = self.training_system.load_checkpoint(str(normalized_path))
            
            if start_epoch is None:
                raise ValueError("Echec du chargement du checkpoint")
            
            await self.broadcast({
                'type': 'checkpoint_loaded',
                'checkpoint': str(normalized_path),
                'start_epoch': start_epoch + 1,
                'best_val_acc': self.training_system.best_val_acc,
                'message': f'Pret a reprendre depuis epoch {start_epoch + 1}',
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Checkpoint charge avec succes: epoch {start_epoch}")
            
        except FileNotFoundError as e:
            logger.error(f"Erreur load_checkpoint: {e}")
            await self.broadcast({
                'type': 'error',
                'message': f'Checkpoint introuvable: {checkpoint_path}'
            })
            self.training_system = None
        except Exception as e:
            logger.error(f"Erreur load_checkpoint: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur chargement checkpoint: {str(e)}'
            })
            self.training_system = None
    
    async def resume_training(self, checkpoint_path: str, epochs: Optional[int] = None):
        """Reprend l'entrainement depuis un checkpoint"""
        try:
            import torch
            
            if self.is_training:
                await self.broadcast({
                    'type': 'error',
                    'message': 'Entrainement deja en cours'
                })
                return
            
            # CORRECTION: Normaliser le chemin
            normalized_path = self._normalize_checkpoint_path(checkpoint_path)
            
            # Verifier l'existence
            if not normalized_path.exists():
                raise FileNotFoundError(f"Checkpoint introuvable: {normalized_path}")
            
            # Charger le checkpoint
            await self.load_checkpoint(str(normalized_path))
            
            if not self.training_system:
                raise ValueError("Systeme d'entrainement non initialise")
            
            # Recuperer l'epoch de depart
            import numpy
            import torch.serialization
            
            with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
                checkpoint_data = torch.load(normalized_path, map_location='cpu', weights_only=False)
            
            start_epoch = checkpoint_data.get('epoch', 0) + 1
            
            # Nombre d'epochs total
            if epochs is None:
                epochs = self.training_system.config.get('training', 'epochs', default=50)
            
            await self.broadcast({
                'type': 'log',
                'message': f'Reprise entrainement epoch {start_epoch} -> {epochs}',
                'level': 'info'
            })
            
            # Lancer l'entrainement
            self.is_training = True
            self.training_task = asyncio.create_task(
                self.training_system.train(epochs=epochs, start_epoch=start_epoch)
            )
            
            logger.info(f"Entrainement repris depuis epoch {start_epoch}")
            
        except FileNotFoundError as e:
            logger.error(f"Erreur resume_training: {e}")
            await self.broadcast({
                'type': 'error',
                'message': f'Checkpoint introuvable: {checkpoint_path}'
            })
            self.is_training = False
        except Exception as e:
            logger.error(f"Erreur resume_training: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur reprise: {str(e)}'
            })
            self.is_training = False
    
    async def delete_checkpoint(self, checkpoint: str):
        """Supprime un checkpoint"""
        try:
            # CORRECTION: Normaliser le chemin
            ckpt_path = self._normalize_checkpoint_path(checkpoint)
            
            if ckpt_path.exists():
                ckpt_path.unlink()
                await self.broadcast({
                    'type': 'log',
                    'message': f'Checkpoint supprime: {ckpt_path.name}',
                    'level': 'success'
                })
                # Renvoyer la liste mise a jour
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
        """Exporte le modele en ONNX"""
        try:
            checkpoint_path = data.get('checkpoint')
            
            # CORRECTION: Normaliser le chemin
            normalized_path = self._normalize_checkpoint_path(checkpoint_path)
            
            # Verifier l'existence
            if not normalized_path.exists():
                raise FileNotFoundError(f"Checkpoint introuvable: {normalized_path}")
            
            if not self.training_system:
                # Creer un systeme temporaire pour l'export
                await self.broadcast({
                    'type': 'log',
                    'message': 'Initialisation pour export...',
                    'level': 'info'
                })
                
                import torch
                import numpy
                import torch.serialization
                
                with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
                    checkpoint_data = torch.load(normalized_path, map_location='cpu', weights_only=False)
                
                # Utiliser la config du checkpoint ou fallback
                if 'config' in checkpoint_data:
                    logger.info("Utilisation de la configuration du checkpoint")
                    saved_config = checkpoint_data['config']
                    from breastai_training import Config, TrainingSystem
                    config = Config(saved_config)
                else:
                    logger.info("Creation configuration depuis metadonnees")
                    architecture = checkpoint_data.get('architecture', 'efficientnetv2_s')
                    num_classes = checkpoint_data.get('num_classes', 3)
                    use_cbam = checkpoint_data.get('use_cbam', True)
                    image_size = checkpoint_data.get('image_size', 512)
                    
                    from breastai_training import Config, TrainingSystem
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
                
                temp_system = TrainingSystem(config, callback=self.broadcast)
                await temp_system.setup()
                
                # Export avec chemin normalise
                success = await temp_system.export_onnx(str(normalized_path))
                
            else:
                # Utiliser le systeme existant
                success = await self.training_system.export_onnx(str(normalized_path))
            
            if success:
                logger.info("Export ONNX reussi")
            else:
                logger.error("Export ONNX echoue")
                
        except FileNotFoundError as e:
            logger.error(f"Erreur export_onnx: {e}")
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
    
    async def system_diagnostics(self):
        """Renvoie les diagnostics systeme"""
        try:
            import psutil
            
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
        """Mappe la config de l'interface vers la structure interne"""
        config = {
            'paths': {
                'data_dir': interface_config.get('data_path', 'data'),
                'checkpoint_dir': interface_config.get('checkpoint_path', 'checkpoints')
            },
            'training': {
                'epochs': interface_config.get('epochs', 50),
                'learning_rate': interface_config.get('learning_rate', 0.0003),
                'weight_decay': interface_config.get('weight_decay', 0.0001),
                'optimizer': interface_config.get('optimizer', 'adamw'),
                'scheduler': interface_config.get('scheduler', 'cosine'),
                'label_smoothing': interface_config.get('label_smoothing', 0.1),
                'gradient_clip': interface_config.get('gradient_clip', 1.0),
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
                'gradient_accumulation_steps': interface_config.get('gradient_accumulation_steps', 1),
                'augmentation': {
                    'horizontal_flip': interface_config.get('horizontal_flip', 0.5),
                    'vertical_flip': interface_config.get('vertical_flip', 0.3),
                    'rotation': interface_config.get('rotation', 15),
                    'brightness': interface_config.get('brightness', 0.2),
                    'contrast': interface_config.get('contrast', 0.2)
                }
            },
            'inference': {
                'tta_enabled': interface_config.get('use_tta', False)
            },
            'model': {
                'architecture': interface_config.get('model_name', 'efficientnetv2_s'),
                'num_classes': interface_config.get('num_classes', 3),
                'use_cbam': interface_config.get('use_cbam', True),
                'dropout_rate': interface_config.get('dropout_rate', 0.4),
                'pretrained': interface_config.get('pretrained', True),
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
    
    async def start(self):
        """Demarre le serveur"""
        logger.info("="*80)
        logger.info("BREASTAI WEBSOCKET SERVER v3.3.1 - PRODUCTION")
        logger.info(f"Demarrage sur ws://{self.host}:{self.port}")
        logger.info("="*80)
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info("Serveur pret! En attente de connexions...")
            await asyncio.Future()

# ==================================================================================
# POINT D'ENTREE
# ==================================================================================

async def main():
    # Creer les dossiers necessaires
    Path('logs').mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    
    # Demarrer le serveur
    server = BreastAIServer(host='localhost', port=8765)
    await server.start()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nServeur arrete par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)