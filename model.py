import pandas as pd
import numpy as np
import time
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import utils

pids = ["BK7610", "BU4707", "CC6740", "DC6359", "HV0618", "JB3156", "JR8022", "MJ8002", "PC6771", "SF3079"]

features = ['mean', 'std', 'median', 'crossing_rate', 'max_abs', 'min_abs', 'max_raw', 'min_raw', 'spec_centroid', 'spec_spread', 'spec_flux', 'rms', 'spec_entrp_freq', 'spec_entrp_time', 'spec_rolloff', 'max_freq']


def separate_dataset():
	data = pd.read_csv("all_accelerometer_data_pids_13.csv")	
	
	for pid in pids:
		print (pid)
		data[data['pid']==pid].to_csv('accelerometer/accelerometer_' + pid +'.csv', index=False)

def create_segments():

	# Segment Data into 10 second long-term segment and 1 second short-term segment
	for pid in pids:
	  	# read the csv file of specific pid
		data = pd.read_csv('accelerometer/accelerometer_' + pid + '.csv')
	    
	    # start with segment 0 i.e. first long-term segment, adding long_segment col for long-term denotion
		long_seg_no = 0
		# start with segment 0 i.e. first short-term segment, adding short_segment col for short-term denotion
		short_seg_no = 0

		data['long_segment'] = long_seg_no
		data['short_segment'] = long_seg_no

		timestamps = data['time']
			
	    # get the starting localtime = timestamp[0]
		local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamps[0]/1000))
	    # convert localtime to datetime format to set delta of 10 seconds (long-term segment)
		start_time = datetime.datetime.strptime(local_time, "%Y-%m-%d %H:%M:%S")
	    
	    # delta of 10 seconds for long segment
		time_step_long = datetime.timedelta(seconds=10)
		# delta of 1 seconds for short segment
		time_step_short = datetime.timedelta(seconds=1)

		start_time_short = start_time
		start_time_long = start_time

		print ('pid: ', pid)
		print (start_time)

		for ind, row in data.iterrows():

	      	# get the timestamp of the record
			local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(row['time']/1000))
			timestamp = datetime.datetime.strptime(local_time, "%Y-%m-%d %H:%M:%S")

			data.at[ind, 'local_time'] = timestamp

		 	# LONG SEGMENTS
	      	# if this is last timestamp in current long-term segment, change the start-time for next segment (long_seg_no++)
			if timestamp == (start_time_long + time_step_long):
				start_time_long += time_step_long
				long_seg_no += 1
				data.at[ind, 'long_segment'] = long_seg_no
				
				# short segment also changes since long segment has changed
				short_seg_no = 0
				start_time_short = start_time_long

	      	# if timestamp is greater then, this timestamp becomes the new start time for next long-term segment (long_seg_no++)
			elif timestamp > (start_time_long + time_step_long):
				start_time_long = timestamp
				long_seg_no += 1
				data.at[ind, 'long_segment'] = long_seg_no
				
				# short segment also changes since long segment has changed
				short_seg_no = 0
				start_time_short = start_time_long
	      
	      	# Still in current long-term segment
			else:
				data.at[ind, 'long_segment'] = long_seg_no


			# SHORT SEGMENTS
			# if still in current short segment
			if timestamp < (start_time_short + time_step_short):
				data.at[ind, 'short_segment'] = short_seg_no
			
			# if timestamp greater than start_time_short, then short segment changes
			else:
				start_time_short += time_step_short
				short_seg_no += 1
				data.at[ind, 'short_segment'] = short_seg_no

		
		print ('long_seg_no: ', long_seg_no)
		data.to_csv('accelerometer/acc_segmented_' + pid + '.csv', index=False)

		print ()

def summary_stats(data, feature, pid):

	global_frame = pd.DataFrame(columns=['Mean_x', 'Mean_y', 'Mean_z', 'Var_x', 'Var_y', 'Var_z', 'Max_x', 'Max_y', 'Max_z', 'Min_x', 'Min_y', 'Min_z', 'Mean_Low_x', 'Mean_Low_y', 'Mean_Low_z', 'Mean_High_x', 'Mean_High_y', 'Mean_High_z', 'Mean_x1', 'Mean_y1', 'Mean_z1', 'Var_x1', 'Var_y1', 'Var_z1', 'Max_x1', 'Max_y1', 'Max_z1', 'Min_x1', 'Min_y1', 'Min_z1', 'Mean_Low_x1', 'Mean_Low_y1', 'Mean_Low_z1', 'Mean_High_x1', 'Mean_High_y1', 'Mean_High_z1'])

	for long_seg in range(len(data)):

		row = []

		lt = np.mean(data[long_seg], axis=0)
		for i in lt:
			row.append(i)
		
		lt = np.var(data[long_seg], axis=0)
		for i in lt:
			row.append(i)

		lt = np.max(data[long_seg], axis=0)
		for i in lt:
			row.append(i)

		lt = np.min(data[long_seg], axis=0)
		for i in lt:
			row.append(i)

		lt = np.mean(np.sort(data[long_seg], axis=0)[:3], axis=0)
		for i in lt:
			row.append(i)

		lt = np.mean(np.sort(data[long_seg], axis=0)[-3:], axis=0)
		for i in lt:
			row.append(i)

		row = np.array(row)

		if long_seg==0:
			for ind, col in enumerate(global_frame.columns):
				if ind<18:
					global_frame.at[long_seg, col] = row[ind]
				else:
					global_frame.at[long_seg, col] = row[ind-18]
		else:
			prev_row = np.array(global_frame.iloc[[long_seg-1]])
			for ind, col in enumerate(global_frame.columns):
				if ind<18:
					global_frame.at[long_seg, col] = row[ind]
				else:
					global_frame.at[long_seg, col] = row[ind-18] - prev_row[0][ind-18]

	# print ('feature: ', feature)
	# print (df)
	print ('      summary_stats done.')
	global_frame.to_csv('features/' + feature + '_feature.csv', index=False)

def extract_dataframes():

	for pid in pids:
		print ()
		print ('pid: ', pid)
		tac_reading = pd.read_csv('clean_tac/' + pid + '_clean_TAC.csv')
		acc_data = pd.read_csv('accelerometer/accelerometer_' + pid + '.csv')

		tac_labels = []

		for feat_no, feature in enumerate(features):
			print ('   feature:', feature)
			array_long = []

			for ind, row in tac_reading.iterrows():
				
				if ind!=0:
				
					t1, t2 = prev_row['timestamp'], row['timestamp']
					long_data = acc_data[ (acc_data['time']/1000 >= t1) & (acc_data['time']/1000 < t2) ]

					if not long_data.empty:
						
						if feat_no==0:
							if prev_row['TAC_Reading'] >= 0.08:
								tac_labels.append(1)
							else:
								tac_labels.append(0) 

						if feature=='rms':
							lt = []
							for axis in ['x', 'y', 'z']:
								lt.append(utils.rms(long_data[axis]))

							lt = np.array(lt)
							array_long.append(lt)

						else:
							short_datas = np.array_split(long_data, 300)
							
							# stores the features for every 1 second in 10 second segment
							array_short = []

							for short_seg, short_data in enumerate(short_datas):

								# data_short = data_long[data_long['short_segment']==short_seg]

								lt = []
								for axis in ['x', 'y', 'z']:
									data_axis =	np.array(short_data[axis])

									if feature=='mean':
										lt.append(utils.mean_feature(data_axis))
									elif feature=='std':
										lt.append(utils.std(data_axis))
									elif feature=='median':
										lt.append(utils.median(data_axis))
									elif feature=='crossing_rate':
										lt.append(utils.crossing_rate(data_axis))
									elif feature=='max_abs':
										lt.append(utils.max_abs(data_axis))
									elif feature=='min_abs':
										lt.append(utils.min_abs(data_axis))
									elif feature=='max_raw':
										lt.append(utils.max_raw(data_axis))
									elif feature=='min_raw':
										lt.append(utils.min_raw(data_axis))
									elif feature=='spec_entrp_freq':
										lt.append(utils.spectral_entropy_freq(data_axis))
									elif feature=='spec_entrp_time':
										lt.append(utils.spectral_entropy_time(data_axis))
									elif feature=='spec_centroid':
										lt.append(utils.spectral_centroid(data_axis))
									elif feature=='spec_spread':
										lt.append(utils.spectral_spread(data_axis))
									elif feature=='spec_rolloff':
										lt.append(utils.spectral_rolloff(data_axis))
									elif feature=='max_freq':
										lt.append(utils.max_freq(data_axis))
									elif feature=='spec_flux':
										if short_seg==0:
											lt.append(utils.spectral_flux(data_axis, np.zeros(len(data_axis))))
											if axis=='x':
												x = data_axis
											elif axis=='y':
												y = data_axis
											elif axis=='z':
												z = data_axis
										else:
											if axis=='x':
												if len(data_axis) > len(x):
													zeros = np.zeros(len(data_axis) - len(x))
													x = np.append(x, zeros)
												elif len(data_axis) < len(x):
													zeros = np.zeros(len(x) - len(data_axis))
													data_axis = np.append(data_axis, zeros)

												lt.append(utils.spectral_flux(data_axis, x))
											elif axis=='y':
												if len(data_axis) > len(y):
													zeros = np.zeros(len(data_axis) - len(y))
													y = np.append(y, zeros)
												elif len(data_axis) < len(y):
													zeros = np.zeros(len(y) - len(data_axis))
													data_axis = np.append(data_axis, zeros)

												lt.append(utils.spectral_flux(data_axis, y))
											elif axis=='z':
												if len(data_axis) > len(z):
													zeros = np.zeros(len(data_axis) - len(z))
													z = np.append(z, zeros)
												elif len(data_axis) < len(z):
													zeros = np.zeros(len(z) - len(data_axis))
													data_axis = np.append(data_axis, zeros)

												lt.append(utils.spectral_flux(data_axis, z))


								array_short.append(np.array(lt))
							
							short_metric = np.array(array_short)
							array_long.append(short_metric)

				prev_row = row
		
			if feature=='rms':
				df = pd.DataFrame(columns=['Rms_x', 'Rms_y', 'Rms_z'])
				long_metric = np.array(array_long)

				df['Rms_x'] = long_metric[:,0:1].flatten()
				df['Rms_y'] = long_metric[:,1:2].flatten()
				df['Rms_z'] = long_metric[:,2:].flatten()

				df.to_csv('features/' + feature + '_feature.csv', index=False)
			else:
				long_metric = np.array(array_long)

				summary_stats(long_metric, feature, pid)
		
		print ('   tac_labels: ', len(tac_labels))
		rename_column_and_concat(pid, tac_labels)

def rename_column_and_concat(pid, tac_labels):

	print ('   Renaming Column Names...')
	for ind, feature in enumerate(features):

		data = pd.read_csv('features/' + feature + '_feature.csv')
		new_cols = []
		for col in data.columns:
			new_cols.append(feature + '_' + col)
		data.columns = new_cols

		print ('      feature: ', feature)
		# data = pd.read_csv('features/' + feature + '_feature.csv')

		if ind!=0:
			data = pd.concat([prev_data, data], axis=1, sort=False)
			# print (data)
		
		prev_data = data

	data['tac_label'] = tac_labels
	data.to_csv('dataset/' + pid + '_data.csv', index=False)

def create_dataset():

	for ind, pid in enumerate(pids):

		data = pd.read_csv('dataset/' + pid + '_data.csv')
		if ind!=0:
			data = pd.concat([data, prev_data], axis=0, sort=False)

		prev_data = data

	data.to_csv('dataset/data.csv', index=False)

def model():
	dataset = pd.read_csv('dataset/data.csv')
	labels = dataset['tac_label']

	data = dataset.to_numpy()
	data = data[:,0:data.shape[1]-1]

	def hyperParamTuning():

		param_grid = {
	    'bootstrap': [True, False],
	    'max_depth': [80, 90, 100, 110],
	    'max_features': [2, 3],
	    'min_samples_leaf': [3, 4, 5],
	    'min_samples_split': [8, 10, 12],
	    'n_estimators': [100, 200, 300, 700, 1000]
		}

		# Create a based model
		rf = RandomForestClassifier()
		# Instantiate the grid search model
		grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 4, n_jobs = -1, verbose=1)

		X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)

		grid_search.fit(X_train, y_train)	
		print (grid_search.best_params_)
		
		best_grid = grid_search.best_estimator_

		y_pred = best_grid.predict(X_test)

		print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
		
	# hyperParamTuning()
	
	rf = RandomForestClassifier(bootstrap =  False, max_depth = 110, max_features = 2, min_samples_leaf = 5, min_samples_split =  8, n_estimators =  700)

	epochs = 100

	accuracies = []

	for e in range(epochs):
		X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state = 43)

		rf.fit(X_train, y_train)
		y_pred = rf.predict(X_test)
		accr = metrics.accuracy_score(y_test, y_pred)
		print ('Accuracy: ', accr)
		accuracies.append(accr)

	print ('Average Accuracy: ', np.mean(accuracies))

def main():

	extract_dataframes()
	create_dataset()
	model()

if __name__ == '__main__':
	main()