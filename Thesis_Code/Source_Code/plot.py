import PySimpleGUI as GUI
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
	#Forming the graph figure.
	if canvas.children:
		for child in canvas.winfo_children():
			child.destroy()
	if canvas_toolbar.children:
		for child in canvas_toolbar.winfo_children():
			child.destroy()
	figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
	figure_canvas_agg.draw()
	toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
	toolbar.update()
	figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)

	
class Toolbar(NavigationToolbar2Tk):
	#Adding the following tools to the toolbar.
	toolitems = [t for t in NavigationToolbar2Tk.toolitems if
				t[0] in ('Home', 'Pan', 'Zoom')]

	def __init__(self, *args, **kwargs):
		super(Toolbar, self).__init__(*args, **kwargs)


def plot_training_results(training_results):

	font = ('Arial', 13) #Font for the layout.

	#Forming the layout for the 'Training Results' GUI window.
	layout = [
			  [GUI.Text('Controls:', font=font)],
			  [GUI.Canvas(key='controls_cv')],
			  [GUI.Text('Figure:', font=font)],
			  [GUI.Column(layout=[[GUI.Canvas(key='fig_cv',size=(400 * 2, 450))]], background_color='#DAE0E6', pad=(0, 0))],
			  [GUI.Button('Plot Graph', font=font, pad=(155,5)), GUI.Button('Continue', font=font, pad=(155,5))],
			 ]

	#Finalizing GUI window creation.
	window = GUI.Window(title='Training Results', layout=layout)
	window.Finalize()

	#Loop for GUI window functions.
	while True:
		#Reading events inside the GUI window.
		event, values = window.Read()

		#Closing the window or pressing 'Continue' button breaks the loop.
		if event == GUI.WIN_CLOSED or event =='Continue':
			window.Close()
			break

		#Pressing 'Plot Graph' button generates the required axis to
		#place the graph.
		if event == 'Plot Graph':
			fig, axs = plt.subplots(2)
			fig.tight_layout(pad=4.0)

			#First axis for accuracy and validation accuracy.
			axs[0].plot(training_results.history["accuracy"], label="train accuracy")
			axs[0].plot(training_results.history["val_accuracy"], label="validation accuracy")
			axs[0].set_ylabel("Accuracy")
			axs[0].set_xlabel("Epoch")
			axs[0].legend(loc="lower right")
			axs[0].set_title("Accuracy Evaluation")

			#Second axis for loss and validation loss.
			axs[1].plot(training_results.history["loss"], label="train error")
			axs[1].plot(training_results.history["val_loss"], label="validation error")
			axs[1].set_ylabel("Error")
			axs[1].set_xlabel("Epoch")
			axs[1].legend(loc="upper right")
			axs[1].set_title("Error Evaluation")

			#Initializing the drawing process together with the given axis.
			draw_figure_w_toolbar(window.FindElement(
				'fig_cv').TKCanvas, fig, window.FindElement('controls_cv').TKCanvas)