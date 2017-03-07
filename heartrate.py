# -*- coding: utf-8 -*

import time, sys

import numpy as np
#import scipy as sp
from scipy import signal
import pandas as pd
import serial, pyfirmata
import cv2

DEBUG = True

HEARTRATE_LINE = 0

LF_MIN = 0.05
LF_MAX = 0.15
HF_MIN = 0.15
HF_MAX = 0.4

PORT = "/dev/cu.usbmodem1411"		# Arduino port: ls /dev/cu*
#PORT = "/dev/cu.usbmodem1421"		# Arduino port: ls /dev/cu*
HR = 0								# Position of Analog-pin into HR-sensor

WINDOW_NAME = "dst"
IMAGE = np.zeros([500, 500, 3], dtype=np.uint8)

WAIT = 10

NOUT = 10
F = np.linspace(LF_MIN, HF_MAX, NOUT)		# 検出したい周波数帯域
L = np.where(F<LF_MAX)[0][-1] + 1			# HFとLFの仕分け用の閾値

LABEL = np.array(["TimeStamp", "BPM", "RRI", "HF", "LF", "HF(%)", "LF(%)"])

def rnd(value, cnm=0) :
	out = round(value, cnm)
	if cnm == 0 :
		out = int(out)
	return out

def stamp() :
	t = time.localtime()
	stamp = [t.tm_hour, t.tm_min, t.tm_sec]
	out = ""
	for i in stamp :
		s = str(i)
		if i < 0 :
			s = "0" + s
		out += s
	return out

class App() :
	def __init__(self) :
		self.bpm = 0					# Beat Per Minute: 心拍回数/min.
		self.rri = 0					# R-R Interval: 心拍の間隔

		self.hf = 0						# 心拍の周波数成分（高周波）：副交感神経
		self.lf = 0						# 心拍の周波数成分（低周波）：交感神経
		self.hf_p = 0.0					# HF比率
		self.lf_p = 0.0					# LF比率

		self.heartrate_time = time.time()
		self.heartrate_flag = False
		self.rri_box = np.array([])		# RRIを貯める
		self.rri_time = np.array([])

		self.box = np.zeros([1, LABEL.shape[0]])

	def arduino_init(self) :		# Arduinoとの接続設定
		print("Arduino init ...")
		self.board = pyfirmata.Arduino(PORT)			# Arduino接続

		it = pyfirmata.util.Iterator(self.board)		# AnalogReadの準備 
		it.start()

		self.hr = self.board.get_pin('a:%s:i' %HR)		# AnalogReadする

		print("Arduino inited")

	def beat(self, calib=False) :				# 心拍センサの値から，BPMとRRIを作る
		value = self.hr.read()

		if value > HEARTRATE_LINE :		
			if self.heartrate_flag == 0 :										# 1回目のセンサ値の変動時のみ，計算実行
				nowtime = time.time()											# 現在時刻
				self.rri = round((nowtime - self.heartrate_time) * 1000, 2)		# RRIを計算(mill second)
				self.heartrate_time = nowtime									# RRI計算用時刻の更新
				self.rri_box = np.append(self.rri_box, self.rri)				# RRIを貯めていく
				self.bpm = rnd(60.0 / (self.rri / 1000.0))						# RRIからBPMを逆算する

				if calib == False :
					self.rri_box = self.rri_box[1:]
					self.psd()

				if DEBUG :
					print("BPM: %s" %self.bpm)
					print("RRI: %s" %self.rri)

			self.heartrate_flag += 1

		else :
			self.heartrate_flag = 0

	def lomb(self) :				# Lomb-ScargleによるPSD計算
		x = np.array([0])										# xは経過時間
		
		for j, i in enumerate(self.rri_box) :
			if j != 0 :
				x = np.append(x, x[-1] + self.rri_box[j])		# [0]からの経過時間を計算して追加していく
		normval = x.shape[0]
		
		y = self.rri_box										# yはRRI

		pgram = signal.lombscargle(x, y, F)						# Fは計測したい周波数帯域
		pgram = np.sqrt(4*(pgram/normval))						# 正規化

		self.lf = np.mean(pgram[:L])							# [L]以下はLF，それ以外はHF
		self.hf = np.mean(pgram[L:])

		self.x = x
		self.y = y
		self.pgram = pgram

	def psd(self) :					# 心拍の周波数成分を分析する
		"""
		#psd = sp.fftpack.fft(self.rri_box)
		#psd = psd * psd
		#psd, fxx, sxx = signal.spectrogram(self.rri_box, fs=0.8, nperseg=self.rri_box.shape[0])
		#psd, fxx = signal.welch(self.rri_box, fs=1.0, window="hanning", nperseg=self.rri_box.shape[0], detrend="linear")
		#self.y = sxx
		#self.x = psd

		hf1 = psd[psd<HF_MIN].shape[0]			# HF成分の数を抽出
		hf2 = psd[psd>HF_MAX].shape[0]
		self.hf = psd.shape[0] - hf1 - hf2

		lf1 = psd[psd<LF_MIN].shape[0]			# LF成分の数を抽出
		lf2 = psd[psd>LF_MAX].shape[0]
		self.lf = psd.shape[0] - lf1 - lf2
		"""

		self.lomb()

		if self.hf == 0 :
			self.hf_p = 0
		else :
			self.hf_p = round(self.hf*1.0 / (self.hf + self.lf), 4)		# HFとLFの比を計算
		if self.lf == 0 :
			self.lf_p = 0
		else :
			self.lf_p = round(self.lf*1.0 / (self.hf + self.lf), 4)

		if DEBUG :
			print("HF: %s" %self.hf)
			print("LF: %s" %self.lf)
			print("HF : LF = %s : %s" %(self.hf_p, self.lf_p))

	def beat_calib(self) :			# 心拍の初期キャリブレーション	
		start = time.time()

		while True :
			self.beat(True)

			if time.time() - start >= 15 :
				break

			time.sleep(WAIT / 1000.0)
		self.rri_box = self.rri_box[-10:]

	def write(self) :
		v = self.box[1:]
		c = LABEL
		df = pd.DataFrame(v, columns=c)
		df.to_csv("heartrate.csv", index=False, encoding="utf-8")

	def main(self) :
		self.arduino_init()

		self.beat_calib()
		self.psd()

		while True :
			self.beat()
			#self.psd()

			cv2.imshow(WINDOW_NAME, IMAGE)
			key = cv2.waitKey(WAIT)
			if key == 27 :
				break

			add = np.array([[stamp(), self.bpm, self.rri, self.hf, self.lf, self.hf_p, self.lf_p]])
			self.box = np.append(self.box, add, axis=0)

		self.write()

		if DEBUG :
			import matplotlib.pyplot as plt
					
			plt.subplot(2, 1, 1)
			plt.plot(self.x, self.y, 'b+')
			plt.subplot(2, 1, 2)
			plt.plot(F, self.pgram)
			plt.show()
			"""
			plt.plot(self.x, self.y, 'b+')
			plt.show()
			"""

		sys.exit("System Exit")

if __name__ == "__main__" :
	print("System Begin")
	app = App()
	app.main()