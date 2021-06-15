import PySimpleGUI as GUI
import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
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

	def on_key_press(event):
		key_press_handler(event, canvas, toolbar)
		canvas.TKCanvas.mpl_connect("key_press_event", on_key_press)
	return
	
class Toolbar(NavigationToolbar2Tk):
	# only display the buttons we need
	toolitems = [t for t in NavigationToolbar2Tk.toolitems if
				t[0] in ('Home', 'Pan', 'Zoom')]
				# t[0] in ('Home', 'Pan', 'Zoom','Save')]

	def __init__(self, *args, **kwargs):
		super(Toolbar, self).__init__(*args, **kwargs)

def plot_history(history):
	layout = [
		# [GUI.T('Graph: y=sin(x)')],
		[GUI.B('Plot'), GUI.B('Continue')],
		[GUI.T('Controls:')],
		[GUI.Canvas(key='controls_cv')],
		[GUI.T('Figure:')],
		[GUI.Column(
			layout=[
				[GUI.Canvas(key='fig_cv',
							# it's important that you set this size
							size=(400 * 2, 400)
							)]
			],
			background_color='#DAE0E6',
			pad=(0, 0)
		)],
		]
	window = GUI.Window(title='Graph with controls', layout=layout)
	window.Finalize()

	while True:
		event, values = window.Read()
		if event == GUI.WIN_CLOSED or event =='Continue':  # always,  always give a way out!
			window.Close()
			break
		if event == 'Plot':
			fig, axs = plt.subplots(2)
			fig.tight_layout(pad=4.0)

			axs[0].plot(history.history["accuracy"], label="train accuracy")
			axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
			axs[0].set_ylabel("Accuracy")
			axs[0].set_xlabel("Epoch")
			axs[0].legend(loc="lower right")
			axs[0].set_title("Accuracy Evaluation")

			axs[1].plot(history.history["loss"], label="train error")
			axs[1].plot(history.history["val_loss"], label="validation error")
			axs[1].set_ylabel("Error")
			axs[1].set_xlabel("Epoch")
			axs[1].legend(loc="upper right")
			axs[1].set_title("Error Evaluation")

			draw_figure_w_toolbar(window.FindElement(
				'fig_cv').TKCanvas, fig, window.FindElement('controls_cv').TKCanvas)