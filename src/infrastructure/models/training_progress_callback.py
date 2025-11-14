"""Custom callback for training progress display."""
import time
from typing import Optional
import tensorflow as tf
from tensorflow import keras


class TrainingProgressCallback(keras.callbacks.Callback):
    """Custom callback to display detailed training progress."""
    
    def __init__(self, total_epochs: int):
        """Initialize the callback.
        
        Args:
            total_epochs: Total number of epochs for training.
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_times = []
    
    def on_train_begin(self, logs=None):
        """Called when training begins."""
        self.start_time = time.time()
        print("\n" + "=" * 70)
        print(f"🚀 Iniciando entrenamiento - {self.total_epochs} épocas")
        print("=" * 70 + "\n")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        epoch_start = time.time()
        self.current_epoch_start = epoch_start
        epoch_num = epoch + 1
        progress = (epoch_num / self.total_epochs) * 100
        
        print(f"\n📊 Época {epoch_num}/{self.total_epochs} ({progress:.1f}%)")
        print("-" * 70)
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        epoch_time = time.time() - self.current_epoch_start
        self.epoch_times.append(epoch_time)
        
        epoch_num = epoch + 1
        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - epoch_num
        estimated_time = avg_time * remaining_epochs
        
        # Format metrics
        loss = logs.get('loss', 0)
        accuracy = logs.get('accuracy', 0) * 100
        
        val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_accuracy')
        
        print(f"  ⏱️  Tiempo: {epoch_time:.1f}s | Promedio: {avg_time:.1f}s/época")
        print(f"  📈 Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%")
        
        if val_loss is not None:
            val_acc_pct = val_accuracy * 100 if val_accuracy else 0
            print(f"  ✅ Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc_pct:.2f}%")
        
        if remaining_epochs > 0:
            mins, secs = divmod(int(estimated_time), 60)
            print(f"  ⏳ Tiempo estimado restante: {mins}m {secs}s")
        
        print()
    
    def on_train_end(self, logs=None):
        """Called when training ends."""
        total_time = time.time() - self.start_time
        mins, secs = divmod(int(total_time), 60)
        
        final_loss = logs.get('loss', 0) if logs else 0
        final_accuracy = logs.get('accuracy', 0) * 100 if logs else 0
        
        print("=" * 70)
        print("✅ Entrenamiento completado!")
        print("=" * 70)
        print(f"  ⏱️  Tiempo total: {mins}m {secs}s")
        print(f"  📊 Loss final: {final_loss:.4f}")
        print(f"  🎯 Accuracy final: {final_accuracy:.2f}%")
        
        if logs and 'val_loss' in logs:
            val_acc = logs.get('val_accuracy', 0) * 100
            print(f"  ✅ Val Accuracy final: {val_acc:.2f}%")
        
        print("=" * 70 + "\n")

