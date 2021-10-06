# built-in libraries
import os
# external libraries
import pandas as pd


def collect_seleced_bstack(csv, outpth, buildmodel, entries):
	print('## collect_selected_bstack.py')
	if buildmodel:
		model_name = entries['Model name']
		outpth = outpth + '/' + model_name
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
	else:
		raise NameError('collect_seleced_bstack is only for buildmodel')
		return

	return df
