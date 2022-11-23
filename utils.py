import numpy as np
import feature_extraction
from scipy.fftpack import fft

def mean_feature(data):
	return np.mean(data)

def std(data):
	return np.std(data)

def median(data):
	return np.median(data)

def crossing_rate(data):

	change = 0

	pos = data[0] >= 0

	for i in range(1, len(data)):
		
		if data[i] < 0 and pos==True:
			change += 1
			pos = False
		elif data[i] > 0 and pos==False:
			change += 1
			pos = True
	
	return change

def max_abs(data):
	return np.max(np.absolute(data))

def min_abs(data):
	return np.min(np.absolute(data))

def max_raw(data):
	return np.max(data)

def min_raw(data):
	return np.min(data)

def spectral_centroid(data):
	fft_magnitude = abs(fft(data))
	sampling_rate = 40
	lt = feature_extraction.spectral_centroid_spread(fft_magnitude, sampling_rate)

	return lt[0]

def spectral_spread(data):
	fft_magnitude = abs(fft(data))
	sampling_rate = 40
	lt = feature_extraction.spectral_centroid_spread(fft_magnitude, sampling_rate)

	return lt[1]

def spectral_entropy_freq(data):
	n_short_blocks = 1
	return feature_extraction.spectral_entropy(abs(fft(data)), n_short_blocks)

def spectral_entropy_time(data):
	n_short_blocks = 1
	return feature_extraction.spectral_entropy(data, n_short_blocks)

def spectral_flux(data, prev_data):
	return feature_extraction.spectral_flux(abs(fft(data)), abs(fft(prev_data)))

def spectral_rolloff(data):
	return feature_extraction.spectral_rolloff(data, 0.9)

def max_freq(data):
	return (np.max(abs(fft(data))))

def rms(data):
   return np.sqrt(np.mean(data ** 2))
