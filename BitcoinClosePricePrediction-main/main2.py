import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QWidget, QFileDialog, QLabel, QScrollArea
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from sklearn import metrics


from scipy import stats

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Toolbar Example")

        # Create actions for the toolbar
        self.action_open = QAction("Open CSV", self)
        self.action_print_info = QAction("Print Dataset Info", self)

        # Connect actions to functions
        self.action_open.triggered.connect(self.open_csv)
        self.action_print_info.triggered.connect(self.print_dataset_info)

        # Create a toolbar
        toolbar = self.addToolBar("Toolbar")
        toolbar.addAction(self.action_open)
        toolbar.addAction(self.action_print_info)

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create layout
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignTop)  # Align the layout to the top

        # Create a horizontal layout for the button and text edit status
        hbox = QHBoxLayout()
        layout.addLayout(hbox)

        # Create a button to import CSV
        self.btn_import_csv = QPushButton("Import CSV")
        self.btn_import_csv.setMaximumWidth(100)  # Adjust the maximum width
        self.btn_import_csv.clicked.connect(self.open_csv)
        hbox.addWidget(self.btn_import_csv)

        # Create QLabel for the status text
        self.label_status = QLabel("Status:")
        self.label_status.setFont(QFont("Arial", 10))  # Set font size
        hbox.addWidget(self.label_status)

        # Create QTextEdit widget for displaying status
        self.text_edit_status = QTextEdit()
        self.text_edit_status.setMaximumWidth(200)  # Adjust the maximum width
        self.text_edit_status.setMaximumHeight(30)  # Adjust the maximum height
        self.text_edit_status.setAlignment(Qt.AlignCenter)  # Set horizontal alignment to centered
        hbox.addWidget(self.text_edit_status)

        # Create QLabel for the info heading
        self.label_info = QLabel("Dataset Information:")
        self.label_info.setFont(QFont("Arial", 10))  # Set font size
        layout.addWidget(self.label_info)

        # Create a QTextEdit widget for displaying dataset information
        self.text_edit_info = QTextEdit()
        self.text_edit_info.setMaximumHeight(200)  # Adjust the maximum height
        layout.addWidget(self.text_edit_info)

        # Create a scroll area for the graph
        self.graph_scroll_area = QScrollArea()
        self.graph_scroll_area.setWidgetResizable(True)
        layout.addWidget(self.graph_scroll_area)

        # Create a widget to contain the graph
        self.graph_widget = QWidget()
        self.graph_scroll_area.setWidget(self.graph_widget)

        # Create a layout for the graph widget
        self.graph_layout = QVBoxLayout(self.graph_widget)

        # Initialize the graph widget
        self.init_graph_widget()

    def init_graph_widget(self):
        # Create FigureCanvas
        self.figure_canvas = FigureCanvas(plt.Figure())

        # Add FigureCanvas to the layout
        self.graph_layout.addWidget(self.figure_canvas)

    def open_csv(self):
        file_dialog = QFileDialog(self)
        filename, _ = file_dialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if filename:
            try:
                dataset = pd.read_csv(filename)
                self.text_edit_status.append("CSV imported successfully.")
                self.dataset = dataset
            except Exception as e:
                self.text_edit_status.append(f"Error loading CSV: {str(e)}")

    def print_dataset_info(self):
     try:
        scaled_data = self.dataset[['Open', 'High', 'Low', 'Close', 'Volume']]
        scaler = MinMaxScaler(copy=False)
        scaled_data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(scaled_data[['Open', 'High', 'Low', 'Close', 'Volume']])
        print(scaled_data)
        
        scaled_data["Up/Down"] = self.dataset["Up/Down"].copy()

        print("crt")
        X = scaled_data[['Open', 'High', 'Low', 'Volume']]
        y = scaled_data['Close']
        lr = LogisticRegression()
        print("yes")
        predicted = cross_val_predict(lr, X, y, cv=6)
        scores = cross_val_score(lr, X, y, scoring='roc_auc', cv=36, n_jobs=1)
        dataset_info_text="Cross-validated predictions:"
        print(predicted)
        print("Cross-validated ROC AUC scores:")
        print(scores)

        linReg = LinearRegression()
        linReg.fit(X, y)  # Train the model on the entire dataset
        future_data = scaler.transform([[self.dataset["Open"].sum(), self.dataset["High"].sum(),
                                          self.dataset["Low"].sum(), self.dataset["Close"].sum(),
                                          self.dataset["Volume"].sum()]])
        future_prediction = linReg.predict(future_data)
        print("Future prediction of Close price:")
        print(future_prediction[0] * -1)

        # Clear existing plots
        self.figure_canvas.figure.clear()

        num_plots = 3  # Number of plots to display per row
        num_graphs = 6  # Total number of graphs to display

        # Calculate the number of rows needed
        num_rows = (num_graphs + num_plots - 1) // num_plots

        # Create subplots
        axes = []
        for i in range(num_graphs):
            ax = self.figure_canvas.figure.add_subplot(num_rows, num_plots, i + 1)
            axes.append(ax)

        # Plot value counts bar plot
        dataset.Private.value_counts().plot.bar(ax=axes[0])
        axes[0].set_title('Private value counts')
        axes[0].set_xlabel('Private')
        axes[0].set_ylabel('Count')

        # Plot line plot
        sns.lineplot(x="Apps", y='Accept', data=dataset, ax=axes[1])
        axes[1].set_title('Line plot')
        axes[1].set_xlabel('Apps')
        axes[1].set_ylabel('Accept')

        # Plot seaborn line plot
        sns.lineplot(x="Enroll", y='Outstate', data=dataset, ax=axes[2])
        axes[2].set_title('Seaborn Line plot')
        axes[2].set_xlabel('Apps')
        axes[2].set_ylabel('Accept')

        male_count = dataset['Male'].sum()
        female_count = dataset['Female'].sum()
        ax = self.figure_canvas.figure.add_subplot(num_rows, num_plots, 4)
        ax.bar('Male', male_count, color='blue', alpha=0.5)
        ax.bar('Female', female_count, color='red', alpha=0.5)
        ax.set_title('Gender Counts')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        ax.legend()
        
        # Plot histogram of 'Grad.Rate'
        ax_hist = self.figure_canvas.figure.add_subplot(num_rows, num_plots, 5)
        ax_hist.hist(dataset['Grad.Rate'], bins=30, color='skyblue', edgecolor='black')
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Histogram of Grad.Rate')

        ax_pie = self.figure_canvas.figure.add_subplot(num_rows, num_plots, 6)
        labels = ['Top10perc', 'Top25perc']
        ax_pie.pie([dataset['Top10perc'].mean(), dataset['Top25perc'].mean()], labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
        ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax_pie.set_title('Pie Chart Example')

        # Adjust layout
        self.figure_canvas.figure.tight_layout()

        # Draw canvas
        self.figure_canvas.draw()

     except AttributeError:
        self.text_edit_status.append("Please import a CSV file first.")

     except AttributeError:
            self.text_edit_status.append("Please import a CSV file first.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
