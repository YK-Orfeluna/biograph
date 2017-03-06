# -*- coding: utf-8 -*

import time

import numpy as np
import scipy as sp
import serial, pyfirmata
import cv2

"""
pyfirmataで，心拍センサとGSR，脳波の値を取得する
心拍センサからは，BPMとRRI
RRIを使って，PSDしてHFとLFを検出
脳波はそこからとれる，フーリエ変換？の結果
GSRのfpsは？？
"""

DEBUG = True

HEARTRATE_LINE = 0.5

LF_MIN = 0.05
LF_MAX = 0.15
HF_MIN = 0.15
HF_MAX = 0.4


PORT = "/dev/cu.usbmodem1411"		# Arduino port: ls /dev/cu*
#PORT = "/dev/cu.usbmodem1421"		# Arduino port: ls /dev/cu*
HR = 0								# Position of Analog-pin into HR-sensor
GSR = 4								# Position of Analog-pin into GSR-sensor

WINDOW_NAME = "dst"
IMAGE = np.zeros([500, 500, 3], dtype=np.uint8)

WAIT = 33

class App() :
	def __init__(self) :
		self.timestamp = 0

		self.bpm = 0					# Beat Per Minute: 心拍回数/min.
		self.rri = 0					# R-R Interval: 心拍の間隔

		self.hf = 0						# 心拍の周波数成分（高周波）：副交感神経
		self.lf = 0						# 心拍の周波数成分（低周波）：交感神経
		self.hf_p = 0.0
		self.lf_p = 0.0

		self.rri_box = np.array([])		# RRIを貯める

		self.gsr = 0

		self.heartrate = 0
		self.heartrate_time = time.time()
		self.heartrate_flag = False

		self.box = np.array([])

		#以下脳波の成分が続く

	def arduino_init(self) :		# Arduinoとの接続設定
		self.board = pyfirmata.Arduino(PORT)			# Arduino接続

		it = pyfirmata.util.Iterator(self.board)		# AnalogReadの準備 
		it.start()

		self.hr = self.board.get_pin('a:%s:i' %HR)		# AnalogReadする
		self.gsr = self.board.get_pin('a:%s:i' %GSR)

		print("Arduino init")

	def beat(self, calib=False) :				# 心拍センサの値から，BPMとRRIを作る
		value = self.hr.read()

		if value > HEARTRATE_LINE :		
			if self.heartrate_flag == 0 :							# 1回目のセンサ値の変動時のみ，計算実行
				nowtime = time.time()								# 現在時刻
				self.rri = (nowtime - self.heartrate_time) * 1000	# RRIを計算(mill second)
				self.heartrate_time = nowtime						# RRI計算用時刻の更新
				self.rri_box = np.append(self.rri_box, self.rri)	# RRIを貯めていく
				self.bpm = 60.0 / (self.rri / 1000.0)				# RRIからBPMを逆算する

				if calib == False :
					self.rri_box = np.append(self.rri_box[1:], self.rri)

				if DEBUG :
					print("BPM: %s" %self.bpm)
					print("RRI: %s" %self.rri)
			
			self.heartrate_flag += 1

		else :
			self.heartrate_flag = 0

		

	def psd(self) :					# 心拍の周波数成分を分析する
		# パワースペクトラムを計算する
		#sp.signal.lombscargle()	# 人間の心拍は一定値ではないので，Lomb-Scargleを使用する
		#fft = np.abs(sp.fftpack.fft(self.rri_box))
		psd = sp.signal.spectrogram(self.rri_box, nperseg=self.rri_box.shape[0])

		hf1 = psd[psd<HF_MIN].shape[0]			# HF成分の数を抽出
		hf2 = psd[psd>HF_MAX].shape[0]
		self.hf = psd.shape[0] - hf1 - hf2

		lf1 = psd[psd<LF_MIN].shape[0]			# LF成分の数を抽出
		lf2 = psd[pad>LF_MAX].shape[0]
		self.lf = psd.shape[0] - lf1 - lf2

		self.hf_p = self.hf*1.0 / (self.hf + self.lf)		# HFとLFの比を計算
		self.lf_p = self.lf*1.0 / (self.hf + self.lf)

	def beat_calib(self) :			# 心拍の初期キャリブレーション	
		start = time.time()

		while True :			# 10秒の間，rriを蓄積させて計算準備
			self.beat(True)

			if self.rri != 0 :
				self.rri_box = np.append(self.rri_box, self.rri)

			if time.time() - start >= 10 :
				break			# 10秒経過したらwhileから抜け出す

	def write(self) :
		label = ""

	def stamp(self) :
		t = time.localtime()
		stamp = [t.tm_hour, t.tm_min, t.tm_sec]
		out = ""
		for i in stamp :
			s = str(i)
			if i < 0 :
				s = "0" + s
			out += s
		self.timestamp = out

	def main(self) :
		self.arduino_init()

		#self.beat_calib()
		#self.psd()

		while True :
			self.beat()
			#print self.hr.read()
			#self.psd()
			#self.gsr()
			#self.brain()

			cv2.imshow(WINDOW_NAME, IMAGE)
			key = cv2.waitKey(WAIT)
			if key == 27 :
				break

			add = np.array([self.timestamp, self.bpm, self.hf, self.lf, self.hf_p, self.lf_p])
			self.box = np.append(self.box, add)


		self.write()
		exit()

if __name__ == "__main__" :
	app = App()
	app.main()