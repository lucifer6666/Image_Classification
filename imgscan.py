import sys
from tkinter import *
from PyQt4 import QtGui,QtCore
import tensorflow as tf
import pyttsx3


class Window(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Window,self).__init__(parent)
        self.setGeometry(150,150,570,100)
        self.setWindowTitle('Image Scanner')
        self.nam=QtGui.QLabel('Image',self)
        self.nam.setGeometry(10,10,60,30)
        self.name=QtGui.QLineEdit('',self)
        self.name.setGeometry(80,10,250,30)
        self.order=QtGui.QLabel('',self)
        self.order.setGeometry(370,10,150,70)
        self.btn=QtGui.QPushButton('Scan',self)
        self.btn.setGeometry(120,50,100,30)
        self.btn.clicked.connect(self.Scan)
        self.img=QtGui.QLabel('',self)
        myPixmap = QtGui.QPixmap('./Data/b.gif')
        myScaledPixmap = myPixmap.scaled(self.order.size())
        self.order.setPixmap(myScaledPixmap)
        self.show()

    def Scan(self):
    	myPixmap = QtGui.QPixmap('./Data/a.gif')
    	myScaledPixmap = myPixmap.scaled(self.order.size())
    	self.order.setPixmap(myScaledPixmap)
    	self.scn()


    def scn(self):
    	image_path=self.name.text()
    	image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    	label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("./Data/output_labels.txt")]
    	with tf.gfile.FastGFile("./Data/output_graph.pb", 'rb') as f:
        	graph_def = tf.GraphDef()
    	graph_def.ParseFromString(f.read())
    	_ = tf.import_graph_def(graph_def, name='')
    	with tf.Session() as sess:
    		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    		predictions = sess.run(softmax_tensor, \
             		{'DecodeJpeg/contents:0': image_data})
    		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    		node_id=top_k[0]
    		human_string = label_lines[node_id]
    		score = predictions[0][node_id]
    		self.order.setText(human_string)
    	
    	self.say()




    def say(self):
    	self.setGeometry(200,200,570,600)
    	self.img.setGeometry(10,100,550,490)
    	image_path=self.name.text()
    	myPixmap = QtGui.QPixmap(image_path)
    	myScaledPixmap = myPixmap.scaled(self.img.size())
    	self.img.setPixmap(myScaledPixmap)
    	t=str(self.order.text())
    	k = pyttsx3.init()
    	rate = k.getProperty('rate')
    	k.setProperty('rate', rate-30)
    	k.runAndWait()
    	k.say('This Is The Image Of '+ t)
    	k.runAndWait()
    	

app=QtGui.QApplication(sys.argv)
GUI=Window()
sys.exit(app.exec_())
