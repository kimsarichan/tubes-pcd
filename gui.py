from Tkinter import Tk, Frame, Menu, Label, BOTH,Button, Scale, IntVar, RIGHT,HORIZONTAL
import Tkinter as tk
import tkFileDialog
import matplotlib, sys
import Image, ImageTk
from ttk import Frame, Style
import cv2
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
import kuantisasi,sampling,histogram,equalization,fliiping,fourier_transform,copy_paste,soal1,blending_soal2,masking,pseudo_color,segementation,morphology
import Image, ImageFilter

class Example(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.parent = parent
        self.filename=""
        self.filename2=""
        self.i=1
        self.j=1
        self.kuan=1
        self.sam=1

        self.initUI()

    def open_image(self,imgs,xs,ys):
        # Load an color image
        img = cv2.imread(str(imgs))
        #Rearrang the color channel
        img=cv2.resize(img,(256,256), interpolation = cv2.INTER_CUBIC)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        # Convert the Image object into a TkPhoto object
        im = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=im)
        label = Label(self, image=imgtk)
        label.image = imgtk
        label.place(x=xs, y=ys)

    def open_withoutsave(self,imgs,xs,ys):
        img=cv2.resize(imgs,(256,256), interpolation = cv2.INTER_CUBIC)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        # Convert the Image object into a TkPhoto object
        im = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=im)
        label = Label(self, image=imgtk)
        label.image = imgtk
        label.place(x=xs, y=ys)

    def openbw(self,imgs,xs,ys):
        img=cv2.resize(imgs,(256,256), interpolation = cv2.INTER_CUBIC)
        # Convert the Image object into a TkPhoto object
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        label = Label(self, image=imgtk)
        label.image = imgtk
        label.place(x=xs, y=ys)

    def browse(self):
        filename = tkFileDialog.askopenfilename( **self.file_opt)
        if filename:
            self.filename=filename
            self.open_image(filename,20,20)
            self.i=1
            self.j=1
            return open(filename,'r')

    def browse2(self):
        filename = tkFileDialog.askopenfilename( **self.file_opt2)
        if filename:
            self.filename2= filename
            self.open_image(filename,300,20)
            return open(filename,'r')

    def rotate(self):
        # Load an color image
        img = cv2.imread(str(self.filename))
        img = cv2.resize(img, (256, 256))
        rows ,cols ,k = img.shape
        a=self.scale.get()
        M = cv2.getRotationMatrix2D((cols/2,rows/2),a,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        self.open_withoutsave(dst,300,20)

    def dilasi(self):
        img = cv2.imread(str(self.filename))
        kernel = np.ones((5,5),np.uint8)
        dilasi = cv2.dilate(img,kernel,iterations = self.j)
        self.open_withoutsave(dilasi,300,20)
        self.j=self.j+1

    def erosi(self):
        img = cv2.imread(str(self.filename))
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(img,kernel,iterations = self.i)
        self.open_withoutsave(erosion,300,20)
        self.i=self.i+1

    def samplings(self):
        self.open_withoutsave(sampling.sampling(str(self.filename),self.scalesamp.get()),300,20)


    def kuantisas(self):
        self.open_withoutsave(kuantisasi.kuantisasi(str(self.filename),self.scalekuan.get()),300,20)

    def zoomout(self):
        img = cv2.imread(str(self.filename))
        height, width = img.shape[:2]
        res = cv2.resize(img,(width/self.scalezoomout.get(), height/self.scalezoomout.get()), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("zoom out",res)

    def zoomin(self):
        img = cv2.imread(str(self.filename))
        height, width = img.shape[:2]
        res = cv2.resize(img,(width*self.scalezoomin.get(), height*self.scalezoomin.get()), interpolation = cv2.INTER_CUBIC)
        cv2.imshow("zoom in",res)

    def onScale(self, val):
        v = int(float(val))
        self.var.set(v)
    def fliphor(self):
        self.open_withoutsave(fliiping.flip_horizontal(str(self.filename)),300,20)
    def flipver(self):
        self.open_withoutsave(fliiping.flip_vertical(str(self.filename)),300,20)
    def cut_paste(self):
        self.open_withoutsave(copy_paste.cut(str(self.filename),str(self.filename2)),300,20)
    def rgb_channel(self):
        soal1.rgb(str(self.filename))
    def fourier_transform(self):
        fourier_transform.fft(str(self.filename))
    def mask(self):
        self.open_withoutsave(masking.mask(str(self.filename),str(self.filename2)),300,20)
    def blend(self):
        blending_soal2.blending(str(self.filename),str(self.filename2))
    def pseudo_col(self):
        self.open_withoutsave(pseudo_color.pseudo_color(str(self.filename)),300,20)
    def rgb_histogram(self):
        img = cv2.imread(str(self.filename))
        color = ('b','g','r')
        b,g,r = cv2.split(img)
        cv2.imshow('image blue ',b)
        cv2.imshow('image red ',r)
        cv2.imshow('image green ',g)
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,128])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
        plt.show()

    def new_window(self):
        t = tk.Toplevel(self)
        img = cv2.imread(str(self.filename))
        img=cv2.resize(img,(128,128), interpolation = cv2.INTER_CUBIC)
        color = ('b','g','r')
        b,g,r = cv2.split(img)
        #blue
        l = tk.Label(t, text="Blue")
        l.place(x=20,y=10)
        im = Image.fromarray(b)
        imgtk = ImageTk.PhotoImage(image=im)
        label = tk.Label(t, image=imgtk)
        label.image = imgtk
        label.place(x=20, y=30)
        #green
        l = tk.Label(t, text="Green")
        l.place(x=168,y=10)
        im = Image.fromarray(g)
        imgtk = ImageTk.PhotoImage(image=im)
        label = tk.Label(t, image=imgtk)
        label.image = imgtk
        label.place(x=168, y=30)
        #red
        l = tk.Label(t, text="Red")
        l.place(x=318,y=10)
        im = Image.fromarray(r)
        imgtk = ImageTk.PhotoImage(image=im)
        label = tk.Label(t, image=imgtk)
        label.image = imgtk
        label.place(x=318, y=30)
        t.geometry("500x200")

    def equalisasi(self):
        equ= equalization.equ(str(self.filename))
        img=cv2.resize(equ,(256,256), interpolation = cv2.INTER_CUBIC)
        img2=cv2.imread(str(self.filename))
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        im = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=im)
        label = Label(self, image=imgtk)
        label.image = imgtk
        label.place(x=300, y=20)
        plt.hist(img2.flatten(),256,[0,256], color = 'b')
        plt.hist(img.flatten(),256,[0,256], color = 'r')
        plt.xlim([0,256])
        plt.show()

    def low_filter_mean(self):
        gam = cv2.imread(str(self.filename))
        blur = cv2.blur(gam,(3,3))
        self.open_withoutsave(blur,300,20)

    def low_filter_median(self):
        gam = cv2.imread(str(self.filename))
        blur = cv2.medianBlur(gam,3)
        self.open_withoutsave(blur,300,20)

    def low_filter_modus(self):
        im=Image.open(str(self.filename))
        im=im.resize((256, 256), Image.ANTIALIAS)
        modus=im.filter(ImageFilter.ModeFilter(3))
        mod = ImageTk.PhotoImage(modus)
        label1 = Label(self, image=mod)
        label1.image = mod
        label1.place(x=300, y=20)
    def mask2(self):
        return

    def laplacian(self):
        img = cv2.imread(str(self.filename),0)
        laplacian = cv2.Laplacian(img,cv2.CV_8U)
        self.openbw(laplacian,300,20)
    def canny(self):
        img = cv2.imread(str(self.filename),0)
        canny=cv2.Canny(img,10,100)
        self.openbw(canny,300,20)
    def prewitt(self):
        img = cv2.imread(str(self.filename),0)
        kernel1 = np.matrix([[-1,0,1],[-1,0,1],[-1,0,1]])
        prewittx = cv2.filter2D(img,-1,kernel1)
        kernel2 = np.matrix([[-1,-1,-1],[0,0,0],[1,1,1]])
        prewitty = cv2.filter2D(img,-1,kernel2)
        prewitt=prewittx+prewitty
        self.openbw(prewitt,300,20)
    def prewitt2(self):
        img = cv2.imread(str(self.filename),0)
        prewwita = np.matrix([[-1,-1,-1],[1,-2,1],[1,1,1]])
        prewwitb =np.matrix([[1,-1,-1],[1,-2,-1],[1,1,1]])
        prewwitc = np.matrix([[1,1,-1],[1,-2,-1],[1,1,-1]])
        prewwitd =np.matrix( [[1,1,1],[1,-2,-1],[1,-1,-1]])
        prewwite = np.matrix([[1,1,1],[1,-2,1],[-1,-1,-1]])
        prewwitf = np.matrix([[1,1,1],[-1,-2,1],[-1,-1,1]])
        prewwitg = np.matrix([[-1,1,1],[-1,-2,1],[-1,1,1]])
        prewwith = np.matrix([[-1,-1,1],[-1,-2,1],[1,1,1]])
        prewitt=cv2.filter2D(img,-1,prewwita)+cv2.filter2D(img,-1,prewwitb)+cv2.filter2D(img,-1,prewwitc)+cv2.filter2D(img,-1,prewwitd)
        prewitt=prewitt+cv2.filter2D(img,-1,prewwite)+cv2.filter2D(img,-1,prewwitf)+cv2.filter2D(img,-1,prewwitg)+cv2.filter2D(img,-1,prewwith)
        self.openbw(prewitt,300,20)

    def sobel(self):
        img = cv2.imread(str(self.filename),0)
        sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
        sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
        sobel=sobelx+sobely
        self.openbw(sobel,300,20)


    def watershed(self):
        self.open_withoutsave(segementation.watershed(str(self.filename)),300,20)
    def hsv_red(self):
        self.open_withoutsave(segementation.hsvred(str(self.filename)),300,20)
    def hsv_blue(self):
        self.open_withoutsave(segementation.hsvblue(str(self.filename)),300,20)
    def hsv_green(self):
        self.open_withoutsave(segementation.hsvgreen(str(self.filename)),300,20)
    def blackwhite(self):
        img = cv2.imread(str(self.filename),0)
        self.openbw(img,20,20)
    def mask_morfo(self):
        self.openbw(morphology.masking(str(self.filename)),300,20)
    def initUI(self):
        self.pack(fill=BOTH, expand=1)
        #zoom in zoom out
        self.btn = Button(self, text="Zoom in",width=10,command=self.zoomin)
        self.btn.place(x=20, y=350)
        self.scalezoomin = Scale(self, from_=1, to=10,orient=HORIZONTAL)
        self.scalezoomin.place(x=105, y=335)
        self.var = IntVar()
        self.label = Label(self, text=0, textvariable=self.var)
        #zoom out
        self.btn = Button(self, text="Zoom out",width=10,command=self.zoomout)
        self.btn.place(x=20, y=390)
        self.scalezoomout = Scale(self, from_=1, to=10,orient=HORIZONTAL)
        self.scalezoomout.place(x=105, y=375)
        self.var = IntVar()
        self.label = Label(self, text=0, textvariable=self.var)
        #dilasi erosi
        self.btn = Button(self, text="Dilasi",width=10, height=1,command=self.dilasi)
        self.btn.place(x=20, y=310)
        self.btn = Button(self, text="Erosi",width=10, height=1,command=self.erosi)
        self.btn.place(x=110, y=310)
        #Kuantisasi
        self.btn = Button(self, text="Kuantisasi",width=10, height=1,command=self.kuantisas)
        self.btn.place(x=300, y=350)
        self.scalekuan = Scale(self, from_=2, to=32,orient=HORIZONTAL)
        self.scalekuan.place(x=385, y=335)
        self.var = IntVar()
        self.label = Label(self, text=0, textvariable=self.var)
        #Sampling
        self.btns = Button(self, text="Sampling",width=10, height=1,command=self.samplings)
        self.btns.place(x=300, y=390)
        self.scalesamp = Scale(self, from_=2, to=8,resolution= 2,orient=HORIZONTAL)
        self.scalesamp.place(x=385, y=375)
        self.var = IntVar()
        self.label = Label(self, text=0, textvariable=self.var)
        #Rotate
        self.btn = Button(self, text="Rotate",width=10, height=1,command=self.rotate)
        self.btn.place(x=300, y=310)
        self.scale = Scale(self, from_=-360, to=360,orient=HORIZONTAL)
        self.scale.place(x=385, y=295)
        self.var = IntVar()
        self.label = Label(self, text=0, textvariable=self.var)

        Style().configure("TFrame")
        self.parent.title("Sotosop")
        # a tk.DrawingArea
        #fig = plt.figure(figsize=(2,2))
        #menu diatas
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
        #file
        fileMenu = Menu(menubar)
        submenu = Menu(fileMenu)
        submenu2 = Menu(fileMenu)
        submenu.add_command(label="Open", command= self.browse)
        #submenu.add_command(label="See name",command=self.save)
        fileMenu.add_cascade(label='Gambar 1', menu=submenu, underline=0)
        submenu2.add_command(label="Open", command= self.browse2)
        #submenu.add_command(label="See name",command=self.save)
        fileMenu.add_cascade(label='Gambar 2', menu=submenu2, underline=0)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", underline=0, command=self.onExit)
        menubar.add_cascade(label="File", underline=0, menu=fileMenu)
        #Edit
        EditMenu= Menu(menubar)
        EditMenu.add_command(label="flip_horizontal", command= self.fliphor)
        EditMenu.add_command(label="flip vertical",command= self.flipver)
        EditMenu.add_command(label="cut and paste",command=self.cut_paste)
        EditMenu.add_command(label="rgb channel",command= self.new_window)
        EditMenu.add_command(label="pseudo color",command=self.pseudo_col)
        EditMenu.add_command(label="fourier transform",command=self.fourier_transform)
        EditMenu.add_command(label="masking",command=self.mask)
        EditMenu.add_command(label="blending",command=self.blend)
        EditMenu.add_command(label="black and white",command=self.blackwhite)
        EditMenu.add_command(label="masking morfology",command=self.mask_morfo)
        menubar.add_cascade(label="Edit",underline=0, menu=EditMenu)
        #histogram
        HistogramMenu = Menu(menubar)
        HistogramMenu.add_command(label="RGB Histogram", command=self.rgb_histogram)
        HistogramMenu.add_command(label="Equalisasi Histogram", command= self.equalisasi)
        menubar.add_cascade(label="Histogram", underline=0, menu=HistogramMenu)
        #segmentasi
        SegmenMenu = Menu(menubar)
        SegmenMenu.add_command(label="Watershed",command=self.watershed)
        submenu_hsv = Menu(SegmenMenu)
        submenu_hsv.add_command(label="Red",command=self.hsv_red)
        submenu_hsv.add_command(label="Green",command=self.hsv_green)
        submenu_hsv.add_command(label="Blue",command=self.hsv_blue)
        SegmenMenu.add_cascade(label="HSV",menu=submenu_hsv)
        menubar.add_cascade(label="Segmentation", underline=0, menu=SegmenMenu)
        #filter

        FilterMenu = Menu(menubar)
        submenu_low = Menu(FilterMenu)
        submenu_high = Menu(FilterMenu)
        submenu_low.add_command(label="Modus Filter",command=self.low_filter_modus)
        submenu_low.add_command(label="Median Filter",command=self.low_filter_median)
        submenu_low.add_command(label="Mean Filter",command=self.low_filter_mean)
        FilterMenu.add_cascade(label="Low Pass",menu=submenu_low)
        submenu_high.add_command(label="Canny Filter",command=self.canny)
        submenu_high.add_command(label="Laplacian Filter",command=self.laplacian)
        submenu_high.add_command(label="Prewitt 1 Filter",command=self.prewitt)
        submenu_high.add_command(label="Prewitt 2 Filter",command=self.prewitt2)
        submenu_high.add_command(label="Sobel Filter",command=self.sobel)
        FilterMenu.add_cascade(label="High Pass",menu=submenu_high)
        menubar.add_cascade(label="Filter Menu", underline=0, menu=FilterMenu)
        #filename
        self.file_opt = options = {}
        self.file_opt2= options ={}

        #image


    def onExit(self):
        self.quit()


def main():

    root = Tk()
    root.geometry("600x450+100+100")
    app = Example(root)
    root.mainloop()


if __name__ == '__main__':
    main()