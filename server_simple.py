#!/usr/bin/env python3
"""
BreastAI WebSocket Server v3.3.0 - SIMPLE ET DIRECT
Serveur WebSocket minimaliste pour contrôler l'entraînement depuis l'interface web
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

# Importer notre système d'entraînement
from breastai_training import TrainingSystem, Config

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
    """Serveur WebSocket simple pour BreastAI"""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.training_system: Optional[TrainingSystem] = None
        self.training_task: Optional[asyncio.Task] = None
        self.is_training = False
        
        logger.info(f"Serveur initialisé sur {host}:{port}")
    
    async def broadcast(self, message: Dict):
        """Envoie un message à tous les clients connectés"""
        if not self.clients:
            return
        
        message_json = json.dumps(message, ensure_ascii=False)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message_json)
            except Exception as e:
                logger.warning(f"Erreur envoi client: {e}")
                disconnected.add(client)
        
        # Nettoyer les clients déconnectés
        self.clients -= disconnected
    
    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Gère une connexion client"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Client connecté: {client_addr}")
        
        try:
            # Message de bienvenue
            await websocket.send(json.dumps({
                'type': 'connection_established',
                'message': 'Connecté au serveur BreastAI',
                'timestamp': datetime.now().isoformat()
            }))
            
            # Boucle de réception
            async for message in websocket:
                await self.handle_message(websocket, message)
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client déconnecté: {client_addr}")
        except Exception as e:
            logger.error(f"Erreur client {client_addr}: {e}", exc_info=True)
        finally:
            self.clients.discard(websocket)
    
    async def handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Traite un message reçu"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            logger.info(f"Message reçu: {msg_type}")
            
            if msg_type == 'start_training':
                # L'interface envoie {type: 'start_training', config: {...}} ou directement les params
                config = data.get('config', data)  # Supporter les deux formats
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
        """Démarre un nouvel entraînement"""
        if self.is_training:
            await self.broadcast({
                'type': 'error',
                'message': 'Entraînement déjà en cours'
            })
            return
        
        try:
            logger.info("Démarrage d'un nouvel entraînement")
            
            # Mapper la config de l'interface
            mapped_config = self._map_interface_config(config_data)
            
            # Créer le système d'entraînement
            config = Config(mapped_config)
            self.training_system = TrainingSystem(config, callback=self.broadcast)
            
            # Setup
            success = await self.training_system.setup()
            if not success:
                await self.broadcast({
                    'type': 'error',
                    'message': 'Échec de l\'initialisation'
                })
                return
            
            # Lancer l'entraînement en arrière-plan
            self.is_training = True
            epochs = config_data.get('epochs', 50)
            
            # Stocker la task pour pouvoir la cancel
            self.training_task = asyncio.create_task(self._run_training(epochs))
            
            logger.info(f"Entraînement lancé: {epochs} epochs")
        
        except Exception as e:
            logger.error(f"Erreur start_training: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur démarrage: {str(e)}'
            })
            self.is_training = False
    
    async def _run_training(self, epochs: int):
        """Exécute l'entraînement"""
        try:
            await self.training_system.train(epochs)
        except Exception as e:
            logger.error(f"Erreur durant l'entraînement: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur entraînement: {str(e)}'
            })
        finally:
            self.is_training = False
            self.training_system = None
    
    async def stop_training(self):
        """Arrête l'entraînement en cours"""
        if not self.is_training or self.training_system is None:
            await self.broadcast({
                'type': 'error',
                'message': 'Aucun entraînement en cours'
            })
            return
        
        logger.info("Arrêt de l'entraînement demandé")
        
        # Arrêter le système
        await self.training_system.stop()
        
        # Cancel la task si elle existe
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
    
    async def list_checkpoints(self):
        """Liste les checkpoints disponibles avec métadonnées"""
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
                    
                    # Essayer de charger les métadonnées
                    try:
                        ckpt_data = torch.load(ckpt_file, map_location='cpu')
                        epoch = ckpt_data.get('epoch', 0)
                        accuracy = ckpt_data.get('best_val_acc', 0.0)
                        architecture = ckpt_data.get('architecture', 'unknown')
                        timestamp = ckpt_data.get('timestamp', '')
                    except Exception:
                        # Si échec lecture métadonnées, valeurs par défaut
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
            
            # Trier par date (plus récent d'abord)
            checkpoints.sort(key=lambda x: x['created'], reverse=True)
            
            await self.broadcast({
                'type': 'checkpoints_list',
                'checkpoints': checkpoints
            })
            
            logger.info(f"Liste de {len(checkpoints)} checkpoints envoyée")
            
        except Exception as e:
            logger.error(f"Erreur list_checkpoints: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur liste checkpoints: {str(e)}'
            })
    
    async def load_checkpoint(self, checkpoint_path: str):
        """Charge un checkpoint et initialise pour reprendre l'entraînement"""
        try:
            await self.broadcast({
                'type': 'log',
                'message': f'Chargement checkpoint: {checkpoint_path}',
                'level': 'info'
            })
            
            # Charger les métadonnées du checkpoint
            import torch
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Récupérer la config du checkpoint
            saved_config = checkpoint_data.get('config', {})
            
            await self.broadcast({
                'type': 'log',
                'message': f'Checkpoint epoch {checkpoint_data.get("epoch", 0)}, accuracy {checkpoint_data.get("best_val_acc", 0):.2f}%',
                'level': 'info'
            })
            
            # Créer le système d'entraînement avec la config sauvegardée
            from breastai_training import Config, TrainingSystem
            config = Config(saved_config)
            self.training_system = TrainingSystem(config, callback=self.broadcast)
            
            # Setup
            await self.broadcast({
                'type': 'log',
                'message': 'Initialisation du système...',
                'level': 'info'
            })
            
            success = await self.training_system.setup()
            
            if not success:
                raise ValueError("Échec du setup")
            
            # Charger le checkpoint
            start_epoch = self.training_system.load_checkpoint(checkpoint_path)
            
            if start_epoch is None:
                raise ValueError("Échec du chargement du checkpoint")
            
            await self.broadcast({
                'type': 'checkpoint_loaded',
                'checkpoint': checkpoint_path,
                'start_epoch': start_epoch + 1,  # Reprendre à l'epoch suivante
                'best_val_acc': self.training_system.best_val_acc,
                'message': f'Prêt à reprendre depuis epoch {start_epoch + 1}',
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Checkpoint chargé avec succès: epoch {start_epoch}")
            
        except Exception as e:
            logger.error(f"Erreur load_checkpoint: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur chargement checkpoint: {str(e)}'
            })
            self.training_system = None
    
    async def resume_training(self, checkpoint_path: str, epochs: Optional[int] = None):
        """Reprend l'entraînement depuis un checkpoint"""
        try:
            import torch
            
            if self.is_training:
                await self.broadcast({
                    'type': 'error',
                    'message': 'Entraînement déjà en cours'
                })
                return
            
            # D'abord charger le checkpoint
            await self.load_checkpoint(checkpoint_path)
            
            if not self.training_system:
                raise ValueError("Système d'entraînement non initialisé")
            
            # Récupérer l'epoch de départ
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            start_epoch = checkpoint_data.get('epoch', 0) + 1
            
            # Nombre d'epochs total (si pas spécifié, continuer jusqu'au max configuré)
            if epochs is None:
                epochs = self.training_system.config.get('training', 'epochs', default=50)
            
            await self.broadcast({
                'type': 'log',
                'message': f'Reprise entraînement epoch {start_epoch} → {epochs}',
                'level': 'info'
            })
            
            # Lancer l'entraînement
            self.is_training = True
            self.training_task = asyncio.create_task(
                self.training_system.train(epochs=epochs, start_epoch=start_epoch)
            )
            
            logger.info(f"Entraînement repris depuis epoch {start_epoch}")
            
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
            ckpt_path = Path(checkpoint)
            if ckpt_path.exists():
                ckpt_path.unlink()
                await self.broadcast({
                    'type': 'log',
                    'message': f'Checkpoint supprimé: {checkpoint}',
                    'level': 'success'
                })
                # Renvoyer la liste mise à jour
                await self.list_checkpoints()
            else:
                await self.broadcast({
                    'type': 'error',
                    'message': f'Checkpoint introuvable: {checkpoint}'
                })
        except Exception as e:
            logger.error(f"Erreur delete_checkpoint: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur suppression: {str(e)}'
            })
    
    async def export_onnx(self, data: Dict):
        """Exporte le modèle en ONNX"""
        try:
            checkpoint_path = data.get('checkpoint')
            
            if not self.training_system:
                # Besoin de créer un système temporaire pour l'export
                await self.broadcast({
                    'type': 'log',
                    'message': 'Initialisation pour export...',
                    'level': 'info'
                })
                
                # Config minimale
                from breastai_training import Config, TrainingSystem
                config = Config()
                temp_system = TrainingSystem(config, callback=self.broadcast)
                
                # Setup du modèle
                await temp_system.setup()
                
                # Export
                success = await temp_system.export_onnx(checkpoint_path)
                
            else:
                # Utiliser le système existant
                success = await self.training_system.export_onnx(checkpoint_path)
            
            if success:
                logger.info("Export ONNX réussi")
            else:
                logger.error("Export ONNX échoué")
                
        except Exception as e:
            logger.error(f"Erreur export_onnx: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur export: {str(e)}'
            })
    
    async def system_diagnostics(self):
        """Renvoie les diagnostics système"""
        import psutil
        
        try:
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
            logger.info("Diagnostics système envoyés")
            
        except Exception as e:
            logger.error(f"Erreur system_diagnostics: {e}", exc_info=True)
            await self.broadcast({
                'type': 'error',
                'message': f'Erreur diagnostics: {str(e)}'
            })
    
    def _map_interface_config(self, interface_config: Dict) -> Dict:
        """Mappe la config de l'interface vers notre structure"""
        config = {
            'paths': {
                'data_dir': interface_config.get('data_path', 'data'),
                'checkpoint_dir': interface_config.get('checkpoint_path', 'checkpoints')
            },
            'model': {
                'architecture': interface_config.get('model_name', 'efficientnetv2_s'),
                'num_classes': interface_config.get('num_classes', 3),
                'use_cbam': interface_config.get('use_cbam', True),
                'dropout_rate': interface_config.get('dropout_rate', 0.4),
                'pretrained': interface_config.get('pretrained', True)
            },
            'training': {
                'epochs': interface_config.get('epochs', 50),
                'learning_rate': interface_config.get('learning_rate', 0.0003),
                'weight_decay': interface_config.get('weight_decay', 0.0001),
                'optimizer': interface_config.get('optimizer', 'adamw'),
                'scheduler': interface_config.get('scheduler', 'cosine'),
                'label_smoothing': interface_config.get('label_smoothing', 0.1),
                'gradient_clip': interface_config.get('gradient_clip', 1.0)
            },
            'data': {
                'batch_size': interface_config.get('batch_size', 4),
                'num_workers': interface_config.get('num_workers', 4),
                'image_size': interface_config.get('image_size', 512),
                'augmentation': {
                    'horizontal_flip': interface_config.get('horizontal_flip', 0.5),
                    'vertical_flip': interface_config.get('vertical_flip', 0.3),
                    'rotation': interface_config.get('rotation', 15),
                    'brightness': interface_config.get('brightness', 0.2),
                    'contrast': interface_config.get('contrast', 0.2)
                }
            }
        }
        
        # Log la config reçue
        logger.info(f"Config mappée: model={config['model']['architecture']}, "
                   f"epochs={config['training']['epochs']}, "
                   f"batch={config['data']['batch_size']}, "
                   f"lr={config['training']['learning_rate']}")
        
        return config
    
    async def start(self):
        """Démarre le serveur"""
        logger.info("="*80)
        logger.info("BREASTAI WEBSOCKET SERVER v3.3.0 - SIMPLIFIE")
        logger.info(f"Démarrage sur ws://{self.host}:{self.port}")
        logger.info("="*80)
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info("Serveur prêt! En attente de connexions...")
            await asyncio.Future()  # Run forever

# ==================================================================================
# POINT D'ENTRÉE
# ==================================================================================

async def main():
    # Créer les dossiers nécessaires
    Path('logs').mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    
    # Démarrer le serveur
    server = BreastAIServer(host='localhost', port=8765)
    await server.start()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nServeur arrêté par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
