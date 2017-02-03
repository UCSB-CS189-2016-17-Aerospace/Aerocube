#import math
#import cv2
#import numpy as np
#import numba
#from numba import cuda
#Sfrom ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker

class MarkerDetectPar:
	

    def _filterDetectedMarkers(cls, corners, ids):
        """
            corners: type vector< vector< Point2f > >
        ids: vector< int >
        """

        # check that corners size is equal to id size
        assert len(corners) == len(id)

        # if corners is empty return 0
        if(len(corners) == 0): return 0

        # remove repeated markers with same id, if one contains the other
        i =0
        while(i < len(corners)):
            j = i +1
            while(j < len(ids)):
                if(ids[i] != ids[j]): 
                    break
                # check if first marker is inside second
                inside = True
                
                for p in range(4):
                    point = corners[j][p]
                    #stil not implemented pointPolygonTest
                    if(pointPolygonTest(corners[i],point,False)<0):
                        inside = False
                        break
                if(inside):
                    toRemove[j] = True
                    atLeastOneRemove = True
                    break
                
                inside = True

                for p in range(4):
                    point = corners[i][p]
                    if(pointPolygonTest(corners[j], point,False)<0):
                        inside = False
                        break
                if(inside):
                    toRemove[i] = True
                    atLeastOneRemove = True
                    break        

                j+=1
            i+=1

        # need to parse the output 




    def pointPolygonTest(cls, corners, thePoint, someBool):
        pass





