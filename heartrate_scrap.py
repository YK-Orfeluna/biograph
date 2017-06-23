# -*- coding: utf-8 -*

import time, sys

import numpy as np
from scipy.signal import lombscargle
import pandas as pd
import cv2
import lxml.html
import requests as rq

DEBUG = True
#DEBUG = False

GSR_FLAG = True
#GSR_FLAG = False

#PLOT = True
PLOT = False

HEARTRATE_LINE = 500

LF_MIN = 0.05
LF_MAX = 0.15
HF_MIN = 0.15
HF_MAX = 0.4

CNT = 10							# RRIの数
CNT *= -1
C_TIME = 10							# キャリブレーションする時間

WINDOW_NAME = "HeartRate & GSR"
IMAGE = np.zeros([500, 500, 3], dtype=np.uint8)
IAMGE.fill(255)

fps = 10

NOUT = 10
F = np.linspace(LF_MIN, HF_MAX, NOUT)		# 検出したい周波数帯域
L = np.where(F<LF_MAX)[0][-1] + 1			# HFとLFの仕分け用の閾値

LED_HR = 4								# 心拍確認用LED
LED = 2									# 動作確認用LED

LABEL = np.array(["Unix", "TimeStamp", "GSR", "BPM", "RRI", "HF", "LF", "HF(%)", "LF(%)"])

TARGET_TXT = "sensorvalue="

def rnd(value, cnm=0) :
	out = round(value, cnm)
	if cnm == 0 :
		out = int(out)
	return out

WAIT = rnd(1000.0 / fps)

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

cnt_p = 0
def printR(out) :
	global cnt_p
	if len(str(out)) < cnt_p :
		cleaner = " " * (cnt_p - len(str(out)))
		sys.stdout.write("\r%s" %out+cleaner)
	else :
		sys.stdout.write("\r%s" %out)
	sys.stdout.flush()
	cnt_p = len(str(out))

class App() :
	def __init__(self, port, url="http://192.168.11.10/") :
		self.port = port
		self.url = url
		
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

	def scraping(self) :
		target_html = rq.get(self.url).text
		root = lxml.html.fromstring(target_html)

		txt = root.text
		txt = txt.lstrip(TARGET_TXT)

		value = ["", ""]
		value = txt.stlip(",")

		self.hr = int(value[0])
		self.galvanic = int(value[1])

	def beat(self, calib=False) :				# 心拍センサの値から，BPMとRRIを作る
		value = self.hr

		if value > HEARTRATE_LINE :		
			if self.heartrate_flag == 0 :										# 1回目のセンサ値の変動時のみ，計算実行
				nowtime = time.time()											# 現在時刻
				self.rri = round((nowtime - self.heartrate_time) * 1000, 2)		# RRIを計算(mill second)
				self.heartrate_time = nowtime									# RRI計算用時刻の更新
				self.rri_box = np.append(self.rri_box, self.rri)				# RRIを貯めていく
				self.bpm = rnd(60.0 / (self.rri / 1000.0))						# RRIからBPMを逆算する

				if calib == False :
					self.rri_box = self.rri_box[1:]								# RRIが一定数になるように調整
					self.psd(calib=calib)

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

		pgram = lombscargle(x, y, F)							# Fは計測したい周波数帯域
		pgram = np.sqrt(4*(pgram/normval))						# 正規化

		self.lf = np.mean(pgram[:L])							# [L]以下はLF，それ以外はHF
		self.hf = np.mean(pgram[L:])

		if DEBUG :
			self.x = x
			self.y = y
			self.pgram = pgram

	def psd(self, calib=True) :					# 心拍の周波数成分を分析する
		self.lomb()

		if self.hf == 0 :
			self.hf_p = 0
		else :
			self.hf_p = round(self.hf*1.0 / (self.hf + self.lf), 4)		# HFとLFの比を計算
		if self.lf == 0 :
			self.lf_p = 0
		else :
			self.lf_p = round(self.lf*1.0 / (self.hf + self.lf), 4)

	def gsr(self) :
		value = self.galvanic
		return value

	def beat_calib(self) :			# 心拍の初期キャリブレーション	
		print("* Calibration")
		start = time.time()

		while True :
			self.scraping()
			self.beat(True)
			self.gsr()

			if time.time() - start >= C_TIME :
				break

			time.sleep(WAIT / 1000.0)
		self.rri_box = self.rri_box[CNT:]

		print("* done calibration")

	def write(self, name="sensing") :
		v = self.box[1:]
		c = LABEL
		df = pd.DataFrame(v, columns=c)
		df.to_csv(name+".csv", index=False, encoding="utf-8")

	def plot(self) :
		import matplotlib.pyplot as plt
		plt.subplot(2, 1, 1)
		plt.plot(self.x, self.y, 'b+')
		plt.subplot(2, 1, 2)
		plt.plot(F, self.pgram)
		plt.show()

	def finish(self) :
		self.write()							# csvに書き出し

		cv2.destroyAllWindows()

		if PLOT :								# 終了時点でのRRIとPSDをプロットする
			self.plot()

		sys.exit("System Exit")

	def main(self) :
		self.beat_calib()

		start_time = time.time()
		csv_flag = 1

		print(LABEL[[0], 2:])

		while True :
			self.scraping()
			self.beat()
			gsr = self.gsr()

			cv2.imshow(WINDOW_NAME, IMAGE)
			key = cv2.waitKey(WAIT)
			if key == 27 :
				self.finish()

			
			add = np.array([[time.time(), stamp(), gsr, self.bpm, self.rri, self.hf, self.lf, self.hf_p, self.lf_p]])
			self.box = np.append(self.box, add, axis=0)
			if DEBUG :
				printR(add[0][2:])

			# To save data-array each 10min.
			now_time = time.time()
			if now_time - start_time >= 60 * 10 :
				self.write(name="sensing"+str(csv_flag))
				start_time = now_time
				csv_flag += 1



if __name__ == "__main__" :
	print("System Begin")
	app = App()
	app.main()