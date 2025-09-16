import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                               QComboBox, QSpinBox, QDoubleSpinBox, QStackedWidget,
                               QGroupBox, QFormLayout, QTextEdit, QFileDialog,
                               QProgressBar, QFrame, QScrollArea)
from PySide6.QtCore import Qt, QThread, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import torch
from torch.utils.data import DataLoader
import os

class TrainingThread(QThread):
    """
    Thread for running model training to prevent GUI freezing
    """
    training_finished = Signal(object, list, list, list)  # model, losses, train_accs, test_accs
    epoch_finished = Signal(int, float, float, float)  # epoch, loss, train_acc, test_acc
    batch_finished = Signal(int, int, float, float)  # epoch, batch, loss, accuracy
    error_occurred = Signal(str)
    
    def __init__(self, model_name, dataset, optimizer_type, learning_rate, 
                 epochs, batch_size, model_custom_name):
        super().__init__()
        self.model_name = model_name
        self.dataset = dataset
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_custom_name = model_custom_name
        
    def run(self):
        try:
            # Import backend and dataset loading function
            from backend import train_model
            from main import get_dataset
            
            # Load actual dataset based on selection
            try:
                train_loader, test_loader, num_classes = get_dataset(self.dataset, self.batch_size)
                print(f"Successfully loaded {self.dataset} dataset with {num_classes} classes")
            except Exception as e:
                self.error_occurred.emit(f"Failed to load dataset {self.dataset}: {str(e)}")
                return
            
            # Define progress callback
            def progress_callback(epoch, loss, train_acc, test_acc):
                self.epoch_finished.emit(epoch, loss, train_acc, test_acc)
            
                        # Train model with progress callback
            def progress_callback(epoch, loss, train_acc, test_acc):
                self.epoch_finished.emit(epoch, loss, train_acc, test_acc)
                
            def batch_callback(epoch, batch, loss, accuracy):
                self.batch_finished.emit(epoch, batch, loss, accuracy)
            
            model, train_losses, train_accuracies, test_accuracies = train_model(
                model_name=self.model_name,
                dataset=self.dataset,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer_type=self.optimizer_type,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                model_custom_name=self.model_custom_name,
                progress_callback=progress_callback,
                batch_callback=batch_callback
            )
            
            self.training_finished.emit(model, train_losses, train_accuracies, test_accuracies)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class TrainingWidget(QWidget):
    """
    Widget for the training interface
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        # 新增：batch级别的数据
        self.batch_losses = []
        self.batch_accuracies = []
        self.batch_indices = []  # 用于记录batch的索引
        self.total_batches_processed = 0  # 全局batch计数器
        self.current_epoch = 0  # 当前epoch计数器
        
        # 初始化batch图表的line对象，避免重复绘制
        self.batch_loss_line = None
        self.batch_accuracy_line = None
        self.ax3_twin = None
        self.ax1_twin = None  # 第一张图的twin axis
        
        self.init_ui()
        
    def init_ui(self):
        # Set modern style
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                color: #212529;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #495057;
            }
            QPushButton {
                background-color: #007bff;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                padding: 6px;
                border: 2px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
                border-color: #007bff;
            }
            QTextEdit {
                border: 2px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                padding: 8px;
            }
            QProgressBar {
                border: 2px solid #ced4da;
                border-radius: 4px;
                text-align: center;
                background-color: #e9ecef;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 2px;
            }
            QLabel {
                color: #495057;
            }
            /* 滚动条样式美化 */
            QScrollBar:vertical {
                background-color: #f8f9fa;
                width: 16px;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
            QScrollBar::handle:vertical {
                background-color: #6c757d;
                border-radius: 7px;
                min-height: 30px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #495057;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #343a40;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;
                border: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                background-color: #f8f9fa;
                height: 16px;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            }
            QScrollBar::handle:horizontal {
                background-color: #6c757d;
                border-radius: 7px;
                min-width: 30px;
                margin: 1px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #495057;
            }
            QScrollBar::handle:horizontal:pressed {
                background-color: #343a40;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;
                border: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        models = ['LeNet', 'LeNet-ReLU', 'LeNet-ReLU-Dropout', 'LeNet-ReLU-Dropout-BN', 
                  'ResNet', 'ResNet18', 'ResNet34', 'AlexNet', 'VGG', 'NiN']
        self.model_combo.addItems(models)
        model_layout.addRow("Model:", self.model_combo)
        
        self.dataset_combo = QComboBox()
        datasets = ['MNIST', 'FashionMNIST', 'EMNIST-byclass', 'EMNIST-bymerge', 
                   'EMNIST-balanced', 'EMNIST-letters', 'EMNIST-digits', 'EMNIST-mnist', 
                   'classify-leaves']
        self.dataset_combo.addItems(datasets)
        model_layout.addRow("Dataset:", self.dataset_combo)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Hyperparameters group
        hyper_group = QGroupBox("Hyperparameters")
        hyper_layout = QFormLayout()
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['Adam', 'SGD'])
        hyper_layout.addRow("Optimizer:", self.optimizer_combo)
        
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.0001, 1.0)
        self.lr_spinbox.setDecimals(4)
        self.lr_spinbox.setValue(0.001)
        self.lr_spinbox.setSingleStep(0.0001)
        hyper_layout.addRow("Learning Rate:", self.lr_spinbox)
        
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 100)
        self.epochs_spinbox.setValue(10)
        hyper_layout.addRow("Epochs:", self.epochs_spinbox)
        
        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 1000)
        self.batch_spinbox.setValue(32)
        self.batch_spinbox.setSingleStep(16)
        hyper_layout.addRow("Batch Size:", self.batch_spinbox)
        
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("Enter custom model name (optional)")
        hyper_layout.addRow("Model Name:", self.model_name_edit)
        
        hyper_group.setLayout(hyper_layout)
        layout.addWidget(hyper_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.train_button = QPushButton("🚀 Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setFixedHeight(40)
        button_layout.addWidget(self.train_button)
        
        self.save_model_button = QPushButton("💾 Save Model")
        self.save_model_button.clicked.connect(self.save_model)
        self.save_model_button.setEnabled(False)
        self.save_model_button.setFixedHeight(40)
        button_layout.addWidget(self.save_model_button)
        
        self.save_logs_button = QPushButton("📊 Save Logs")
        self.save_logs_button.clicked.connect(self.save_logs)
        self.save_logs_button.setEnabled(False)
        self.save_logs_button.setFixedHeight(40)
        button_layout.addWidget(self.save_logs_button)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Training status
        self.status_label = QLabel("Ready to train")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #28a745;")
        layout.addWidget(self.status_label)
        
        # Training logs
        log_group = QGroupBox("Training Logs")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Training visualization
        viz_group = QGroupBox("Training Visualization")
        viz_layout = QVBoxLayout()
        
        # Set matplotlib style for better appearance (with fallback)
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                plt.style.use('default')
                
        # 创建三张图：第一张训练loss+accuracy、第二张epoch级别accuracies对比、第三张batch级别连续曲线
        # 缩小图表尺寸到原来的0.75倍：14*0.75=10.5, 18*0.75=13.5
        self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10.5, 13.5))
        self.figure.patch.set_facecolor('white')
        
        self.canvas = FigureCanvas(self.figure)
        
        # 相应调整画布的最小尺寸：1200*0.75=900, 1600*0.75=1200
        self.canvas.setMinimumSize(900, 1200)
        viz_layout.addWidget(self.canvas)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # 创建主内容widget并将所有内容添加到其中
        content_widget = QWidget()
        content_widget.setLayout(layout)
        
        # 创建整个页面的滚动区域
        main_scroll_area = QScrollArea()
        main_scroll_area.setWidgetResizable(True)
        main_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll_area.setWidget(content_widget)
        
        # 设置主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(main_scroll_area)
        
        self.setLayout(main_layout)
        
    def start_training(self):
        model_name = self.model_combo.currentText()
        dataset = self.dataset_combo.currentText()
        optimizer_type = self.optimizer_combo.currentText()
        learning_rate = self.lr_spinbox.value()
        epochs = self.epochs_spinbox.value()
        batch_size = self.batch_spinbox.value()
        model_custom_name = self.model_name_edit.text()
        
        if not model_custom_name:
            model_custom_name = f"{model_name}_{dataset}"
        
        self.status_label.setText("🔄 Initializing training...")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffc107;")
        self.train_button.setEnabled(False)
        self.train_button.setText("Training...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, epochs)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # Clear previous training data
        # Initialize training data storage
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.batch_losses = []
        self.batch_accuracies = []
        self.batch_indices = []
        self.total_batches_processed = 0  # 重置全局batch计数器
        self.current_epoch = 0  # 重置当前epoch计数器
        
        # 初始化batch图表的line对象，避免重复绘制
        self.batch_loss_line = None
        self.batch_accuracy_line = None
        self.ax3_twin = None
        
        # 重置第一张图的twin axis
        self.ax1_twin = None
        
        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.canvas.draw()
        
        self.log_text.append(f"Starting training: {model_name} on {dataset}")
        self.log_text.append(f"Hyperparameters: LR={learning_rate}, Epochs={epochs}, Batch={batch_size}")
        self.log_text.append("-" * 50)
        
        # Start training in a separate thread
        self.training_thread = TrainingThread(
            model_name, dataset, optimizer_type, learning_rate, 
            epochs, batch_size, model_custom_name
        )
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.epoch_finished.connect(self.on_epoch_finished)
        self.training_thread.batch_finished.connect(self.on_batch_finished)
        self.training_thread.error_occurred.connect(self.on_training_error)
        self.training_thread.start()
        
    def on_batch_finished(self, epoch, batch, loss, accuracy):
        """Handle batch completion for batch-level visualization"""
        # 检查是否进入新的epoch，如果是则重置当前epoch计数
        if epoch != self.current_epoch:
            self.current_epoch = epoch
        
        # 使用真正的全局batch索引（连续递增）
        self.total_batches_processed += 1
        
        self.batch_losses.append(loss)
        self.batch_accuracies.append(accuracy)
        self.batch_indices.append(self.total_batches_processed)
        
        # 每15个batch更新一次batch级别的图表，减少UI更新频率（从10改为15提高性能）
        if len(self.batch_losses) % 15 == 0:
            self.update_batch_plots()
            
        # 添加batch级别的日志（每30个batch记录一次，减少日志频率提高性能）
        if batch % 30 == 0:
            self.log_text.append(f"  Epoch {epoch}, Batch {batch}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")
            # Auto-scroll to bottom
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def on_epoch_finished(self, epoch, loss, train_acc, test_acc):
        """Handle epoch completion for real-time updates"""
        self.train_losses.append(loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        
        self.progress_bar.setValue(epoch)
        self.status_label.setText(f"🔄 Training... Epoch {epoch} completed")
        self.log_text.append(f"Epoch {epoch}: Loss={loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
        
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        
        # Update plots in real-time
        self.update_plots()
    
    def on_training_finished(self, model, train_losses, train_accuracies, test_accuracies):
        self.model = model
        # Ensure we have the complete data
        if len(train_losses) > len(self.train_losses):
            self.train_losses = train_losses
            self.train_accuracies = train_accuracies
            self.test_accuracies = test_accuracies
        
        self.status_label.setText("✅ Training completed successfully!")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #28a745;")
        self.train_button.setEnabled(True)
        self.train_button.setText("🚀 Start Training")
        self.progress_bar.setVisible(False)
        self.save_model_button.setEnabled(True)
        self.save_logs_button.setEnabled(True)
        
        self.log_text.append("-" * 50)
        self.log_text.append("🎉 Training completed successfully!")
        if train_accuracies:
            self.log_text.append(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
        if test_accuracies:
            self.log_text.append(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
        
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        
        # Final plot update
        self.update_plots()
        
    def on_training_error(self, error_msg):
        self.status_label.setText("❌ Training error occurred!")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #dc3545;")
        self.train_button.setEnabled(True)
        self.train_button.setText("🚀 Start Training")
        self.progress_bar.setVisible(False)
        self.log_text.append("-" * 50)
        self.log_text.append(f"❌ Error: {error_msg}")
        
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        
    def update_plots(self):
        if not self.train_losses:
            return
            
        # Clear epoch-level plots
        self.ax1.clear()
        self.ax2.clear()
        
        # 清除第一张图的twin axis（如果存在）
        if self.ax1_twin is not None:
            self.ax1_twin.remove()
        
        epochs = list(range(1, len(self.train_losses) + 1))
        
        # Plot 1: 训练集的loss和accuracy曲线（epoch级别）- 使用双Y轴
        self.ax1_twin = self.ax1.twinx()
        
        # 绘制loss（左Y轴）
        line1 = self.ax1.plot(epochs, self.train_losses, linewidth=2.5, marker='o', 
                             markersize=4, label='Training Loss', color='#007bff')
        self.ax1.set_xlabel('Epoch', fontsize=11)
        self.ax1.set_ylabel('Loss', fontsize=11, color='#007bff')
        self.ax1.tick_params(axis='y', labelcolor='#007bff')
        
        # 绘制训练accuracy（右Y轴）
        line2 = []
        if self.train_accuracies:
            line2 = self.ax1_twin.plot(epochs, self.train_accuracies, linewidth=2.5, marker='s',
                                 markersize=4, label='Training Accuracy', color='#28a745')
            self.ax1_twin.set_ylabel('Accuracy (%)', fontsize=11, color='#28a745')
            self.ax1_twin.tick_params(axis='y', labelcolor='#28a745')
        
        self.ax1.set_title('Training Loss & Accuracy (Epoch-level)', fontsize=12, fontweight='bold')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_facecolor('#f8f9fa')
        
        # 添加图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        self.ax1.legend(lines, labels, loc='upper left', fontsize=10)
        
        # Plot 2: 训练集和测试集accuracy对比（epoch级别）
        if self.train_accuracies:
            self.ax2.plot(epochs, self.train_accuracies, linewidth=2.5, marker='o',
                         markersize=4, label='Training Accuracy', color='#28a745')
        if self.test_accuracies:
            self.ax2.plot(epochs, self.test_accuracies, linewidth=2.5, marker='s',
                         markersize=4, label='Test Accuracy', color='#dc3545')
        self.ax2.set_xlabel('Epoch', fontsize=11)
        self.ax2.set_ylabel('Accuracy (%)', fontsize=11)
        self.ax2.set_title('Training vs Test Accuracy Comparison (Epoch-level)', fontsize=12, fontweight='bold')
        self.ax2.legend(fontsize=10)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_facecolor('#f8f9fa')
        
        # Plot 3: Batch-level training progress
        self.update_batch_plots()
        
        # Adjust layout and refresh with optimization
        try:
            self.figure.tight_layout(pad=4.0, h_pad=3.0)
        except:
            pass  # 忽略布局警告，避免影响性能
        self.canvas.draw_idle()  # 使用draw_idle提高响应性
        
    def update_batch_plots(self):
        """Update batch-level plots with optimization and no overlap"""
        if not self.batch_losses:
            return
            
        # 采样优化：只在数据量较大时才进行采样，减少绘图点数提高性能
        if len(self.batch_losses) > 1000:
            step = max(1, len(self.batch_losses) // 500)
            plot_indices = self.batch_indices[::step]
            plot_losses = self.batch_losses[::step]
            plot_accuracies = self.batch_accuracies[::step]
        else:
            plot_indices = self.batch_indices
            plot_losses = self.batch_losses
            plot_accuracies = self.batch_accuracies
        
        # 检查是否需要初始化图表
        if self.batch_loss_line is None:
            # 首次初始化batch图表
            self.ax3.clear()
            if self.ax3_twin is not None:
                self.ax3_twin.remove()
            
            self.ax3_twin = self.ax3.twinx()
            
            # 创建空的line对象
            self.batch_loss_line, = self.ax3.plot([], [], linewidth=1.2, 
                                                  label='Training Loss', color='#007bff', alpha=0.8)
            self.batch_accuracy_line, = self.ax3_twin.plot([], [], linewidth=1.2,
                                                           label='Training Accuracy', color='#dc3545', alpha=0.8)
            
            # 设置标签和样式
            self.ax3.set_xlabel('Batch (Global Index)', fontsize=11)
            self.ax3.set_ylabel('Loss', fontsize=11, color='#007bff')
            self.ax3.tick_params(axis='y', labelcolor='#007bff')
            self.ax3_twin.set_ylabel('Accuracy (%)', fontsize=11, color='#dc3545')
            self.ax3_twin.tick_params(axis='y', labelcolor='#dc3545')
            self.ax3.set_title('Training Progress (Batch-level - Continuous)', fontsize=12, fontweight='bold')
            self.ax3.grid(True, alpha=0.3)
            self.ax3.set_facecolor('#f8f9fa')
            
            # 添加图例
            lines = [self.batch_loss_line, self.batch_accuracy_line]
            labels = [l.get_label() for l in lines]
            self.ax3.legend(lines, labels, loc='upper left', fontsize=10)
        
        # 更新line数据而不是重新绘制
        self.batch_loss_line.set_data(plot_indices, plot_losses)
        self.batch_accuracy_line.set_data(plot_indices, plot_accuracies)
        
        # 自动调整坐标轴范围（添加安全检查避免警告）
        if plot_indices and len(plot_indices) > 1:
            x_min, x_max = min(plot_indices), max(plot_indices)
            # 确保x轴范围不相等
            if x_max > x_min:
                self.ax3.set_xlim(x_min, x_max)
            
            if plot_losses and len(plot_losses) > 0:
                loss_min, loss_max = min(plot_losses), max(plot_losses)
                loss_range = loss_max - loss_min
                # 防止范围为0的情况
                if loss_range > 0:
                    self.ax3.set_ylim(loss_min - 0.1 * loss_range, loss_max + 0.1 * loss_range)
                else:
                    # 如果所有loss值相同，设置一个合理的范围
                    center_val = loss_min
                    self.ax3.set_ylim(center_val - 0.1, center_val + 0.1)
            
            if plot_accuracies and len(plot_accuracies) > 0:
                acc_min, acc_max = min(plot_accuracies), max(plot_accuracies)
                acc_range = acc_max - acc_min
                # 防止范围为0的情况
                if acc_range > 0:
                    self.ax3_twin.set_ylim(acc_min - 0.1 * acc_range, acc_max + 0.1 * acc_range)
                else:
                    # 如果所有accuracy值相同，设置一个合理的范围
                    center_val = acc_min
                    self.ax3_twin.set_ylim(max(0, center_val - 5), min(100, center_val + 5))
        
        # 只更新第三个子图，不重新布局整个figure
        self.canvas.draw_idle()  # 使用draw_idle代替draw提高性能
        
    def save_model(self):
        if self.model is None:
            self.log_text.append("No trained model to save!")
            return
            
        model_name = self.model_combo.currentText()
        dataset = self.dataset_combo.currentText()
        custom_name = self.model_name_edit.text()
        
        if not custom_name:
            save_name = f"{model_name}_{dataset}"
        else:
            save_name = custom_name
            
        # Ensure models/classic_model directory exists
        models_dir = os.path.join("models", "classic_model")
        os.makedirs(models_dir, exist_ok=True)
        
        # Create filename with timestamp if file already exists
        base_path = os.path.join(models_dir, f"{save_name}.pth")
        file_path = base_path
        
        if os.path.exists(file_path):
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(models_dir, f"{save_name}_{timestamp}.pth")
        
        try:
            torch.save(self.model.state_dict(), file_path)
            self.log_text.append(f"✅ Model saved successfully to: {file_path}")
            
            # Also save model info
            info_path = file_path.replace('.pth', '_info.txt')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"Model Information\n")
                f.write("=" * 30 + "\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Dataset: {dataset}\n")
                f.write(f"Optimizer: {self.optimizer_combo.currentText()}\n")
                f.write(f"Learning Rate: {self.lr_spinbox.value()}\n")
                f.write(f"Epochs: {self.epochs_spinbox.value()}\n")
                f.write(f"Batch Size: {self.batch_spinbox.value()}\n")
                if self.train_accuracies:
                    f.write(f"Final Training Accuracy: {self.train_accuracies[-1]:.2f}%\n")
                if self.test_accuracies:
                    f.write(f"Final Test Accuracy: {self.test_accuracies[-1]:.2f}%\n")
                    
            self.log_text.append(f"   - Model info: {info_path}")
            
        except Exception as e:
            self.log_text.append(f"❌ Error saving model: {str(e)}")
            
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
            
    def save_logs(self):
        if not self.train_losses:
            self.log_text.append("No training data to save!")
            return
            
        # Create model-specific folder in logs/classic_model
        model_name = self.model_combo.currentText()
        dataset = self.dataset_combo.currentText()
        custom_name = self.model_name_edit.text()
        
        if not custom_name:
            folder_name = f"{model_name}_{dataset}"
        else:
            folder_name = custom_name
            
        # 使用指定的路径 logs\classic_model
        base_log_dir = os.path.join("logs", "classic_model")
        os.makedirs(base_log_dir, exist_ok=True)
        
        # 创建模型特定的文件夹
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_folder_name = f"{folder_name}_{timestamp}"
        full_log_dir = os.path.join(base_log_dir, full_folder_name)
        
        try:
            os.makedirs(full_log_dir, exist_ok=True)
            
            # 保存三张训练图片
            fig_path = os.path.join(full_log_dir, "training_plots.png")
            self.figure.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            # 保存详细的训练日志
            log_path = os.path.join(full_log_dir, "training_log.txt")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"CNN Training Platform - Training Log\n")
                f.write("=" * 50 + "\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Dataset: {dataset}\n")
                f.write(f"Custom Name: {custom_name if custom_name else 'None'}\n")
                f.write("\nTraining Hyperparameters:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Optimizer: {self.optimizer_combo.currentText()}\n")
                f.write(f"Learning Rate: {self.lr_spinbox.value()}\n")
                f.write(f"Epochs: {self.epochs_spinbox.value()}\n")
                f.write(f"Batch Size: {self.batch_spinbox.value()}\n")
                f.write(f"Training Start Time: {timestamp}\n")
                f.write("\n" + "=" * 50 + "\n")
                f.write("Training Progress Log:\n")
                f.write("-" * 50 + "\n")
                f.write(self.log_text.toPlainText())
                
            # 保存训练指标为CSV (epoch级别)
            import pandas as pd
            epoch_metrics_path = os.path.join(full_log_dir, "epoch_metrics.csv")
            epoch_data = {
                'Epoch': list(range(1, len(self.train_losses) + 1)),
                'Loss': self.train_losses,
                'Train_Accuracy': self.train_accuracies,
                'Test_Accuracy': self.test_accuracies
            }
            df_epoch = pd.DataFrame(epoch_data)
            df_epoch.to_csv(epoch_metrics_path, index=False)
            
            # 保存batch级别的指标（如果有数据）
            if self.batch_losses:
                batch_metrics_path = os.path.join(full_log_dir, "batch_metrics.csv")
                batch_data = {
                    'Batch_Index': self.batch_indices,
                    'Loss': self.batch_losses,
                    'Accuracy': self.batch_accuracies
                }
                df_batch = pd.DataFrame(batch_data)
                df_batch.to_csv(batch_metrics_path, index=False)
            
            # 保存训练超参数为JSON
            import json
            params_path = os.path.join(full_log_dir, "hyperparameters.json")
            params = {
                "model": model_name,
                "dataset": dataset,
                "custom_name": custom_name,
                "optimizer": self.optimizer_combo.currentText(),
                "learning_rate": self.lr_spinbox.value(),
                "epochs": self.epochs_spinbox.value(),
                "batch_size": self.batch_spinbox.value(),
                "timestamp": timestamp
            }
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
                
            self.log_text.append(f"✅ Logs saved successfully to: {full_log_dir}")
            self.log_text.append(f"   - Training plots: training_plots.png")
            self.log_text.append(f"   - Training log: training_log.txt")
            self.log_text.append(f"   - Epoch metrics: epoch_metrics.csv")
            if self.batch_losses:
                self.log_text.append(f"   - Batch metrics: batch_metrics.csv")
            self.log_text.append(f"   - Hyperparameters: hyperparameters.json")
            
        except Exception as e:
            self.log_text.append(f"❌ Error saving logs: {str(e)}")
            
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

class CustomModelWidget(QWidget):
    """
    Placeholder widget for custom model training
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Create a modern "coming soon" card
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 2px solid #e9ecef;
                padding: 40px;
            }
        """)
        card_layout = QVBoxLayout(card)
        
        # Icon and title
        icon_label = QLabel("🔧")
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(icon_label)
        
        title_label = QLabel("Custom Model Training")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #495057;
            margin: 20px 0;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title_label)
        
        desc_label = QLabel("This feature is coming soon!\nYou'll be able to create and train your own custom neural network architectures.")
        desc_label.setStyleSheet("""
            font-size: 14px;
            color: #6c757d;
            line-height: 1.5;
        """)
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        card_layout.addWidget(desc_label)
        
        layout.addWidget(card)
        layout.addStretch()
        self.setLayout(layout)

class MainWindow(QMainWindow):
    """
    Main application window
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 CNN Model Training Platform")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set modern application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QPushButton#sidebar_button {
                background-color: #ffffff;
                border: 2px solid #e9ecef;
                color: #495057;
                padding: 12px;
                border-radius: 8px;
                font-weight: bold;
                text-align: left;
                margin: 2px;
            }
            QPushButton#sidebar_button:hover {
                background-color: #007bff;
                color: white;
                border-color: #007bff;
            }
            QPushButton#sidebar_button:checked {
                background-color: #007bff;
                color: white;
                border-color: #007bff;
            }
            QWidget#sidebar {
                background-color: #ffffff;
                border-right: 2px solid #e9ecef;
            }
        """)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create sidebar
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("sidebar")
        sidebar_widget.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(15, 20, 15, 20)
        sidebar_layout.setSpacing(10)
        
        # App title in sidebar
        title_label = QLabel("🤖 CNN Training")
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 20px;
            padding: 10px;
        """)
        sidebar_layout.addWidget(title_label)
        
        # Navigation buttons
        self.classic_btn = QPushButton("📊 Classic Models")
        self.classic_btn.setObjectName("sidebar_button")
        self.classic_btn.setCheckable(True)
        self.classic_btn.setChecked(True)
        self.classic_btn.clicked.connect(self.show_classic_training)
        sidebar_layout.addWidget(self.classic_btn)
        
        self.custom_btn = QPushButton("🔧 Custom Models")
        self.custom_btn.setObjectName("sidebar_button")
        self.custom_btn.setCheckable(True)
        self.custom_btn.clicked.connect(self.show_custom_training)
        sidebar_layout.addWidget(self.custom_btn)
        
        sidebar_layout.addStretch()
        
        # Version info
        version_label = QLabel("v1.0.0")
        version_label.setStyleSheet("""
            color: #6c757d;
            font-size: 10px;
            margin-top: 10px;
        """)
        sidebar_layout.addWidget(version_label)
        
        # Create stacked widget for main content
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("""
            QStackedWidget {
                background-color: #f8f9fa;
                border: none;
            }
        """)
        
        self.training_widget = TrainingWidget()
        self.custom_widget = CustomModelWidget()
        
        self.stacked_widget.addWidget(self.training_widget)
        self.stacked_widget.addWidget(self.custom_widget)
        
        # Add widgets to main layout
        main_layout.addWidget(sidebar_widget)
        main_layout.addWidget(self.stacked_widget)
        
    def show_classic_training(self):
        self.stacked_widget.setCurrentIndex(0)
        self.classic_btn.setChecked(True)
        self.custom_btn.setChecked(False)
        
    def show_custom_training(self):
        self.stacked_widget.setCurrentIndex(1)
        self.classic_btn.setChecked(False)
        self.custom_btn.setChecked(True)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()