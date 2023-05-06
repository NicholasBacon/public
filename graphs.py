import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from statistics import mean
from statistics import median
import statsmodels.api as sm
from enum import Enum

import os, glob









def loggpPingPong(k, L, o, g, G, P):
    # return (o+(k-1)*G+(P-2)*(L+o+g+(k-1)*G)+L+o)
    return (2 * o + (P - 1) * (L + (k / 2 - 1) * G) + (P - 2) * (g + o)) / 1e6
def loggpPingPong(k, L, o_s,o_r,  g, G, P):
    # return (o+(k-1)*G+(P-2)*(L+o+g+(k-1)*G)+L+o)
    return (o_s+o_r + (L + (k/2 - 1) * G) ) / 1e6*2+g/1e6



def getData(folder_path,pingpong,logGP):
    for filename in list(glob.glob(os.path.join(folder_path, '*'))):
        with open(filename, 'r') as f:
            text = f.read()
            key = filename.split("runs\\")[1]
            pingpong[key] = {}

            logGP[key] = {}
            for x in text.split("\n"):
                y = x.split(" ")
                if (len(y) == 33):
                    keyvalue = x.split(",    ")

                    pingpong[key][int(keyvalue[0])] = float(keyvalue[-4])

                if 'L='  in x:
                    print(y)
                    sizekey = int(y[2].split("=")[1])
                    logGP[key][(sizekey)] = {}
                    for logGPTHING in y:

                        keyvalue = logGPTHING.split("=")
                        if len(keyvalue) == 2:
                            logGP[key][sizekey][keyvalue[0]] = float(keyvalue[1])





def getValues(os_s, or_s, xs, ys):
    A1 = np.vstack([xs, np.ones(len(os_s))]).T
    o_s, oo1 = np.linalg.lstsq(A1, np.array(os_s), rcond=None)[0]
    A2 = np.vstack([xs, np.ones(len(or_s))]).T
    o_r, oo2 = np.linalg.lstsq(A2, np.array(or_s), rcond=None)[0]
    A2 = np.vstack([xs, np.ones(len(xs))]).T
    a, b = np.linalg.lstsq(A2, np.array(ys), rcond=None)[0]

    # print(1-np.linalg.lstsq(A2, np.array(or_s), rcond=None)[1][0]/sum((np.array(or_s)- mean(or_s) )**2)  )

    return o_s, oo1, o_r, oo2, a, b

class Graphs(Enum):
    ErrorPlot = 1
    TimePlot =2
    absERR0R = 3
    nonabsERR0R = 4
    oGraphs =5



def makgraphs(pingpong,logGP,rang,PLOT,ERR0R):
    for keys in pingpong.keys():

        xs = list(sorted(map(lambda x: int(x), pingpong[keys].keys())))

        xsSets = set(xs)
        xssSets = set(logGP[keys].keys())

        xs = list(sorted(filter(lambda x: x in rang, xsSets.intersection(xssSets))))
        os_s = list(map(lambda x: logGP[keys][x]['o_s'], xs))
        or_s = list(map(lambda x: logGP[keys][x]['o_r'], xs))
        # 3,2 3,3
        if len(list(map(lambda x: logGP[keys][x]['g'], xs)))  >1    :
            L = min(filter(lambda x: x == x, map(lambda x: logGP[keys][x]['L'], xs)))

            g = max(filter(lambda x: x == x, map(lambda x: logGP[keys][x]['g'], xs)))
            G = max(filter(lambda x: x == x, map(lambda x: logGP[keys][x]['G'], xs)))

            P = 3
            ys = list(map(lambda x: float(pingpong[keys][x]), xs))



            o_s, oo1, o_r, oo2, a, b = getValues(os_s, or_s, xs, np.array(ys))



            if PLOT == Graphs.oGraphs:
                # plt.plot(np.array(xs),
                #           o_r,
                #          '-', label='loggp', markersize=3, color='green')
                #
                # plt.plot(np.array(xs),
                #           o_s,
                #          '-', label='loggp', markersize=3, color='green')
                # draw lines
                xmin = 1
                xmax = 9
                y = 5
                height = 1

                # plt.hlines(y, xmin, xmax)
                # plt.vlines(xmin, y - height / 2., y + height / 2.)
                # plt.vlines(xmax, y - height / 2., y + height / 2.)

                # draw a point on the line
                px = 4
                print(np.array(list(filter(lambda x: x == x, map(lambda x: logGP[keys][x]['g'], xs)))))
                plt.plot(  np.array(list(filter(lambda x: logGP[keys][x]['g'] == logGP[keys][x]['g'],  xs))),  (np.array(list(filter(lambda x: x == x, map(lambda x: logGP[keys][x]['g'], xs)))))  , 'ro', ms=15, mfc='r')

                # add an arrow
                # plt.annotate('Price five days ago', (px, y), xytext=(px - 1, y + 1),
                #              arrowprops=dict(facecolor='black', shrink=0.1),
                #              horizontalalignment='right')



            if PLOT == Graphs.TimePlot:
                # plt.plot(np.array(xs),np.array(xs)*a+b, '-', label=keys, markersize=3, color='blue')


                plt.plot(np.array(xs),np.array(ys), '-.', label='real', markersize=3, color='red')
                plt.plot(np.array(xs),  loggpPingPong(np.array(xs), L,
                                                      (np.array(xs) * o_r) , (
                                                              np.array(xs) * o_s) ,
                                                      g, a,
                                                      P), '-', label='loggp', markersize=3, color='green')

            if PLOT ==Graphs.ErrorPlot:

                if ERR0R== Graphs.absERR0R:

                    plt.plot(np.array(xs),
                             100 * abs(np.array(ys)   -loggpPingPong(np.array(xs), L,
                                                                     (np.array(xs) * o_r) , (
                                                                             np.array(xs) * o_s) ,
                                                                     g, a,
                                                                     P)) / (np.array(ys) ) , '--', markersize=3,
                             )
                    plt.plot(np.array(xs),
                             100 * abs(np.array(ys) - loggpPingPong(np.array(xs), L,
                                                                    (np.array(xs) * o_r) , (
                                                                            np.array(xs) * o_s) ,
                                                                    g, a,
                                                                    P)) / (np.array(ys) ), '*', label=keys, markersize=3,
                             )

                elif   ERR0R== Graphs.nonabsERR0R:
                    print(a,o_r,a-o_r)
                    plt.plot(np.array(xs),
                             100 * (np.array(ys)   - loggpPingPong(np.array(xs), L,
                                                                   (np.array(xs) * o_r), (
                                                                           np.array(xs) * o_s),
                                                                   g, a,
                                                                   P)) / (np.array(ys) ), '--', markersize=3,
                             )
                    plt.plot(np.array(xs),
                             100 * (np.array(ys) - loggpPingPong(np.array(xs), L,
                                                                 (np.array(xs) * o_r) , (
                                                                         np.array(xs) * o_s) ,
                                                                 g, a,
                                                                 P)) / (np.array(ys) ), '*', label=keys, markersize=3,
                             )



def makeGraph(folder_path, logy, logx ,listofranges,graphtype,errortype,allnewplots):

    pingpong = {}
    logGP = {}
    getData(folder_path,pingpong,logGP)
    for rang in listofranges:
        makgraphs(pingpong, logGP, rang ,graphtype,errortype)

    if logy:
        plt.yscale('symlog', base=10)
    if logx:
        plt.xscale('log', base=2)
    if allnewplots:
        plt.title("graph of  prodected times and ping pong of " + folder_path.split("\\")[-1] + "")

        plt.xlabel("Bytes")

        plt.ylabel("times seconds")
    # plt.legend()

        plt.show()

folder_path = 'runs\\mpich'
allnewplots = False
# makeGraph('runs\\mpich', False, True ,[range(2**0,2**8+1)], Graphs.oGraphs,Graphs.nonabsERR0R,allnewplots)
# makeGraph('runs\\mvap2', False, True ,[range(2**0,2**8+1)], Graphs.oGraphs,Graphs.nonabsERR0R,allnewplots)
makeGraph('runs\\spec++', False, True ,[range(2**0,2**8+1)],Graphs.oGraphs,Graphs.nonabsERR0R,allnewplots)
makeGraph('runs\\open', False, True ,[range(2**0,2**8+1)],  Graphs.oGraphs,Graphs.nonabsERR0R,allnewplots)
if not allnewplots:
    plt.title("graph of  prodected times and ping pong of " + folder_path.split("\\")[-1] + "")

    plt.xlabel("Bhytes")

    plt.ylabel("times sehconds")
    plt.show()
# makgraphs(pingpong, logGP, range(2**0,2**8+1) ,Graphs.ErrorPlot,Graphs.nonabsERR0R)

# makgraphs(pingpong, logGP, range(2**0,2**19) ,Graphs.ErrorPlot,Graphs.absERR0R)























