import cv
  
def single_channel_hist( channel ):
    """ calculate cumulative histgram of single image channel
        return as a list of length 256
    """
    hist = cv.CreateHist([256],cv.CV_HIST_ARRAY,[[0,256]],1)
    cv.CalcHist([cv.GetImage(channel)],hist)
    refHist = [cv.QueryHistValue_1D(hist,i) for i in range(0,256)]
    def add(x,y):return x+y
    sum = reduce(add,refHist[:])
    pdf = [v/sum for v in refHist]
    for i in range(1,256):
        pdf[i] = pdf[i-1] + pdf[i]
    return pdf
 
def get_channels(img):
    """split jpg image file into 3 seperate channels
        return as a list of length 3
    """
    channels = []
    for i in range(0,3):
        _refCh = cv.CreateMat(img.rows, img.cols, cv.CV_8UC1)
        channels.append(_refCh)
    cv.Split(img,channels[0],channels[1],channels[2],None)
    return channels
 
def cal_hist(channels):
    """
        cal cumulative hist for channel list
    """
    return [single_channel_hist(channel) for channel in channels]
 
def cal_trans(ref,adj):
    """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
    """
    i =0
    j = 0;
    table = range(0,256)
    for i in range( 1,256):
        for j in range(1,256):
            if ref[i] >= adj[j-1]  and ref[i] <= adj[j]:
                table[i] = j
                break
         
    table[255] = 255
    return table
 
 
if __name__ == '__main__':
      
    cv.NamedWindow('reference')
    cv.NamedWindow('before')
    cv.NamedWindow('after')
 
    refImg = cv2.imread('me.jpg')
    dstImg = cv2.imread('lena.jpg')
     
    refChannels = get_channels(refImg)
    dstChannels = get_channels(dstImg)
 
    cv.ShowImage('reference',refImg);
    cv.ShowImage('before',dstImg);
 
    hist_ref = cal_hist(refChannels)
    hist_dst = cal_hist(dstChannels)
 
    tables = [cal_trans(hist_dst[i],hist_ref[i]) for i in range(0,3)]
 
    for i in range(0,3):
        for j in range(0,dstChannels[i].rows):
            for k in range(0,dstChannels[i].cols):
                v = dstChannels[i][j,k]
                dstChannels[i][j,k] = tables[i][int(v)]
    cv.Merge(dstChannels[0],dstChannels[1],dstChannels[2],None,dstImg)
 
    cv.ShowImage('after',dstImg)
    cv.WaitKey(0)
