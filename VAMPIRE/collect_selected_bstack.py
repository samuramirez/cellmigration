# built-in libraries
import os
# external libraries
import pandas as pd
from tkinter import END


def collect_seleced_bstack(csv, outpth, buildmodel, entries):
	print('## collect_selected_bstack.py')
	if buildmodel:
		ui = pd.read_csv(csv)
		setpaths = ui['set location']
		tag = ui['tag']
		condition = ui['condition']
		bstacks = []
		for setidx, setpath in enumerate(setpaths):
			pickles = [_ for _ in os.listdir(outpth) if _.lower().endswith('pickle')]
			bstack = [pd.read_pickle(os.path.join(outpth, pkl)) for pkl in pickles if condition[setidx] in pkl]
			bstacks = bstacks + bstack
		try:
			df = pd.concat(bstacks, ignore_index=True)
		except:
			print('CSV file is empty')
			df = None
			entries['Status'].delete(0, END)
			entries['Status'].insert(0, 'CSV file is empty')
	else:
		raise NameError('collect_seleced_bstack is only for buildmodel')
		return

	return df
