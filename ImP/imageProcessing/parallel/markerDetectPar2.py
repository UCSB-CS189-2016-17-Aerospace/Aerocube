import math
import cv2
import numpy as np
import numba
from numba import cuda
from ImP.fiducialMarkerModule.fiducialMarker import FiducialMarker

class MarkerDetectPar:

    def detectMarkers(self):
        pass
        # Step 1
        # Step 2
        # Step 3
        # self._filterDetectedMarkers()
        # below is yet to do
        # _copyVector2Output()
        #  Mat(ids).copyTo(_ids);
        # Step 4



    def _filterDetectedMarkers(self, corners, ids):
        """
        corners: type vector< vector< Point2f > >
        ids: vector< int >
        """

        # check that corners size is equal to id size, not sure if assert is done correctly
        assert len(corners) == len(id)

        if len(corners) == 0:
            return 0

        # mark markers that will be deleted, initializes array all set to false ?
        toRemove = [False] * len(corners)
        atLeastOneRemove = False

        # remove repeated markers with same id, if one contains the other
        for i in range(len(corners)-1):
            for j in range(1, len(corners)):
                if ids[i] == ids[j]:
                    return 0

                # check if first marker is inside second
                inside = True
                for p in range(4):
                    point = corners[j][p]
                    if cv2.pointPolygontest(corners[i], point, False) < 0:
                        inside = False
                        break

                if inside:
                    toRemove[j] = True
                    atLeastOneRemove = True
                    continue

                # check the second marker
                inside = True
                for p in range(4):
                    point = corners[i][p]
                    if cv2.pointPolygonTest(corners[j], point, False) < 0:
                        inside = False
                        break

                if inside:
                    toRemove[i] = True
                    atLeastOneRemove = True
                    continue

        if atLeastOneRemove:
            filteredCorners = corners
            filteredIds = ids
            place_hold = 0
            for i in range(len(toRemove)):
                if not toRemove[i]:
                    filteredCorners[i+1] = corners[i]
                    filteredIds[i+1] = ids[i]
                    place_hold = i+1
            '''
            Below is my attempt at recreating the following c++ code:
            _ids.erase(filteredIds, _ids.end());
            _corners.erase(filteredCorners, _corners.end());
            Not sure if I understood it correctly.
            '''
            for element in range(place_hold, len(corners)):
                del filteredCorners[element]

            for element in range(place_hold, len(ids)):
                del filteredIds[element]
