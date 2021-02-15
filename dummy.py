class rectangle ():
    def __init__(self,   width, height ):
        self.height = height
        self.width = width
    def area(self):
        return  self.height*self.width

    def permiter (self):
        return  2*(self.height+self.width)


class square(rectangle):
    def __init__(self, w):
        self.w = w
        super().__init__( self.w, self.w)

class wonder_rectangle (rectangle):
    def __init__(self, w, h):
        self.w = w
        self.h = h
        super().__init__(2*w,2*h)
        super().__init__(w,  h)


    def info (self):
        print (self.w, self.h)
        print (self.width,self.height)

    # def permiter(self):
    #     print (self.width, self.height, super(wonder_rectangle, self).premiter())
    #     print (self.w, self.h, 2*(self.w+self.h))





if __name__ == '__main__':
    # s = square(w= 4.5)
    # print ('area %.2f preimeter %.2f'%(s.area(),s.premiter()))

    w_r = wonder_rectangle(2,3)
    w_r.info()
    print (w_r.permiter())



