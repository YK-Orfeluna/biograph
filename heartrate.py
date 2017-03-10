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

CNT = 10							# RRIの数
CNT *= -1
C_TIME = 15							# キャリブレーションする時間

PORT = "/dev/cu.usbmodem1411"		# Arduino port: ls /dev/cu*
#PORT = "/dev/cu.usbmodem1421"		# Arduino port: ls /dev/cu*

HR = 0								# Position of Analog-pin into HR-sensor
GSR = 2								# Position of Analog-pin into GSR-sensor

WINDOW_NAME = "dst"
IMAGE = np.zeros([500, 500, 3], dtype=np.uint8)

WAIT = 10

NOUT = 10
F = np.linspace(LF_MIN, HF_MAX, NOUT)		# 検出したい周波数帯域
L = np.where(F<LF_MAX)[0][-1] + 1			# HFとLFの仕分け用の閾値

LED_HR = 4								# 心拍確認用LED
LED = 2									# 動作確認用LED

LABEL = np.array(["TimeStamp", "GSR", "BPM", "RRI", "HF", "LF", "HF(%)", "LF(%)"])

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
		self.hr = 0						# sensor-value
		self.galvanic = 0

		self.bpm = 0					# Beat Per Minute: 心拍回数/min.
		self.rri = 0					# R-R Interval: 心拍の間隔

		self.hf = 0						# 心拍の周波数成分（高周波）：副交感神経
		self.lf = 0						# 心拍の周波数成分（低周波）：交感神経
		self.hf_p = 0.0					# HF比率
		self.lf_p = 0.0					# LF比率

		self.heartrate_time = time.time()
		self.heartrate_flag = False
		self.rri_box = np.array([])		# RRIを貯める

		self.box = np.zeros([1, LABEL.shape[0]])

	def arduino_init(self) :		# Arduinoとの接続設定
		print("Arduino init ...")
		self.board = pyfirmata.Arduino(PORT)			# Arduino接続

		it = pyfirmata.util.Iterator(self.board)		# AnalogReadの準備 
		it.start()

		self.hr = self.board.get_pin('a:%s:i' %HR)		# AnalogReadする
		self.galvanic = self.board.get_pin('a:%s:i' %GSR)

		self.board.digital[LED].write(1)				# Digital-pin[2]をhighに
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
					self.rri_box = self.rri_box[1:]								# RRIが一定数になるように調整
					self.psd()

				if DEBUG :
					print("BPM: %s" %self.bpm)
					print("RRI: %s" %self.rri)

				self.board.digital[LED_HR].write(1)

			self.heartrate_flag += 1

		else :
			self.heartrate_flag = 0
			self.board.digital[LED_HR].write(0)

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

		if DEBUG :
			self.x = x
			self.y = y
			self.pgram = pgram

	def psd(self) :					# 心拍の周波数成分を分析する
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

	def gsr(self) :
		value = self.galvanic.read() * 1023

		if DEBUG and self.heartrate_flag == 1:
			print("GSR: %s" %value)

		return value

	def beat_calib(self) :			# 心拍の初期キャリブレーション	
		start = time.time()

		while True :
			self.beat(True)
			self.gsr()

			if time.time() - start >= C_TIME :
				break

			time.sleep(WAIT / 1000.0)
		self.rri_box = self.rri_box[CNT:]

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
			gsr = self.gsr()

			cv2.imshow(WINDOW_NAME, IMAGE)
			key = cv2.waitKey(WAIT)
			if key == 27 :
				break

			if self.heartrate_flag == 1 :
				add = np.array([[stamp(), gsr, self.bpm, self.rri, self.hf, self.lf, self.hf_p, self.lf_p]])
				self.box = np.append(self.box, add, axis=0)

		self.board.digital[LED].write(0)
		self.board.digital[LED_HR].write(0)

		self.write()			# csvに書き出し

		if DEBUG :				# 終了時点でのRRIとPSDをプロットする
			import matplotlib.pyplot as plt
					
			plt.subplot(2, 1, 1)
			plt.plot(self.x, self.y, 'b+')
			plt.subplot(2, 1, 2)
			plt.plot(F, self.pgram)
			plt.show()

		sys.exit("System Exit")

if __name__ == "__main__" :
	print("System Begin")
	app = App()
	app.main()