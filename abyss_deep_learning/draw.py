"""
Module for visualizing various machine learning outputs.

"""

import numpy as np
import cv2
from skimage import color as skic
from skimage import segmentation as skis

def masks(labels, image=None, fill=True, border=False, colors=None, alpha=0.3, image_alpha=1, bg_label=-1, bg_color=(0, 0, 0), kind='overlay'):
    """Draws mask on image or just mask, if image not present
    
    todo
    ----
        review usage
        review performance (currently some 5 seconds for 1860x1240 image)
    
    parameters
    ----------
        labels: see skimage.color.labels2rgb() for details
        image: see skimage.color.labels2rgb() for details
        fill: bool
            fill areas 
        border: bool
            draw border
        colors: see skimage.color.labels2rgb() for details
        alpha: see skimage.color.labels2rgb() for details
        image_alpha: see skimage.color.labels2rgb() for details
        bg_label: see skimage.color.labels2rgb() for details
        bg_color: see skimage.color.labels2rgb() for details
        kind: see skimage.color.labels2rgb() for details
    
    returns
    -------
    np.ndarray
        RGB base-256 image
        
    example
    -------
        import cv2
        import numpy as np
        import abyss_deep_learning as adl
        from abyss_deep_learning import draw

        image = cv2.imread( 'some.image.jpg' )
        labels = np.fromfile("labels.bin",dtype=np.float32)
        labels = np.reshape( labels, ( 1240, 1860 ) )
        drawn = adl.draw.masks( labels, image, image_alpha=0.7, border=True, bg_label=0, colors=('orange', 'white') )
        cv2.imshow( 'drawn', drawn )
        cv2.waitKey( 0 )
        cv2.destroyAllWindows()
    """
    masked = image
    if fill:
        f = skic.label2rgb( labels, colors=colors, alpha=1, bg_label=bg_label, bg_color=bg_color, kind=kind )
        f = ( f * 255 ).astype( np.uint8 )
        f = f[..., ::-1]
        masked = f if masked is None else cv2.addWeighted( masked, image_alpha, f, alpha, 0 )
    if border:
        b = skic.label2rgb( labels * skis.find_boundaries(labels), colors=colors, alpha=1, bg_label=bg_label, bg_color=bg_color )
        b = ( b * 255 ).astype( np.uint8 )
        b = b[..., ::-1]
        masked = b if masked is None else cv2.addWeighted( masked, image_alpha, b, 1, 0 )
    return masked

def boxes(labels, image, fill=False, border=False, colors=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255]], alpha=0.3, image_alpha=1, thickness=1):
    """Draw boxes on an image
    
    todo
    ----
        ? labels: separate boxes and labels? represent as tuples? (may be yet another performance hit)
        ? reconcile ski.color and cv2 colors (something like # todo? use_skimage_color_dict = type(colors[0]) is str?)
        ? is it worth to keep border parameter? (added just for compatibility to masks(), which probably is unimportant
        ? bg_color=(0, 0, 0)
        ? draw border and fill separately (just like in masks()?
        ? if image not given, draw on blank canvas (but then need to pass image dimensions)
        ? refactor using skimage.color.set_color and skimage.color.rectangle
        ! review performance
    
    parameters
    ----------
        labels: np.ndarray
            cols*rows*5 array where every 5-tuple represents rectangle start coordinates, rectangle end coordinates, and label
        image: np.ndarray
            image
        fill: bool
            fill rectangle, same as thickness < 0
        border: bool
            draw rectangle border
        colors: list of r,g,b values
            [[255,255,0],
             [255,0,255]
            ]
        alpha: float
            [0,1]: alpha for rectangles
        image_alpha: float
            [0,1]: alpha for image
        thickness: int
            line thickness, thickness < 0 is same as fill set to True
    
    returns
    -------
    np.ndarray
        RGB base-256 image
    
    example
    -------
        import cv2
        import numpy as np
        import abyss_deep_learning as adl
        from abyss_deep_learning import draw

        image = cv2.imread( 'some.image.jpg' )
        drawn = adl.draw.boxes( [[200,200,300,400,0],[500,600,800,900,1]], drawn, fill=True, colors=('darkorange', 'green'), image_alpha=0.7 )
        drawn = adl.draw.boxes( [[200,200,300,400,0],[500,600,800,900,1]], drawn, colors=('darkorange', 'green'), alpha=1 )
        cv2.imshow( 'drawn', drawn )
        cv2.waitKey( 0 )
        cv2.destroyAllWindows()
    """
    if fill: thickness = -1
    mask = np.zeros((len(image),len(image[0]),3), dtype=np.uint8)
    for label in labels:  # quick and dirty, watch performance
        c = colors[label[4]%len(colors)]
        cv2.rectangle(mask, ( int(label[0]),  int(label[1])), (int(label[2]), int(label[3])), (int(c[2]), int(c[1]), int(c[0])), thickness)
    return cv2.addWeighted( image, image_alpha, mask, alpha, 0 )

def polygons(labels, image, fill=False, border=False, colors=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255]], alpha=0.3, image_alpha=1, thickness=1):
    """Draw boxes on an image
    parameters
    ----------
        labels: np.ndarray
            cols*rows*2 array where every 2-tuple represents segmentation points and label
        image: np.ndarray
            image
        fill: bool
            fill rectangle, same as thickness < 0
        border: bool
            draw rectangle border
        colors: list of r,g,b values
            [[255,255,0],
             [255,0,255]
            ]
        alpha: float
            [0,1]: alpha for rectangles
        image_alpha: float
            [0,1]: alpha for image
        thickness: int
            line thickness, thickness < 0 is same as fill set to True
    
    returns
    -------
    np.ndarray
        RGB base-256 image
    
    example - TODO NOT CORRECT
    -------
        import cv2
        import numpy as np
        import abyss_deep_learning as adl
        from abyss_deep_learning import draw

        image = cv2.imread( 'some.image.jpg' )
        drawn = adl.draw.polygons( [[200,200,300,400],0],[500,600,800,900],1]], drawn, fill=True, colors=('darkorange', 'green'), image_alpha=0.7 )
        drawn = adl.draw.polygons( [[200,200,300,400],0],[500,600,800,900],1]], drawn, colors=('darkorange', 'green'), alpha=1 )
        cv2.imshow( 'drawn', drawn )
        cv2.waitKey( 0 )
        cv2.destroyAllWindows()
    """
    mask = np.zeros((len(image),len(image[0]),3), dtype=np.uint8)
    for label in labels:  # quick and dirty, watch performance
        c = colors[label[1]%len(colors)]
        point_set = label[0]
        point_set_list = np.array([point_set[n:n+2] for n in range(0, len(point_set), 2)],np.int32)
        point_set_list.reshape((-1,1,2))
        point_set_list.astype(dtype=int)
        cv2.polylines(mask,[point_set_list], True,(int(c[2]), int(c[1]), int(c[0])),thickness)
        
        if fill:
            cv2.fillPoly(mask,[point_set_list],(int(c[2]), int(c[1]), int(c[0])))
    return cv2.addWeighted( image, image_alpha, mask, alpha, 0 )





def text(labels, image, colors=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255]],font=cv2.FONT_HERSHEY_SIMPLEX, alpha=0.3, image_alpha=1, scale=1,thickness=1):
    mask = np.zeros((len(image),len(image[0]),3), dtype=np.uint8)
    for label in labels:  # quick and dirty, watch performance
        c = colors[label[3]%len(colors)]
        cv2.putText(mask,label[2],(int(label[0]),int(label[1])),font,scale,(int(c[2]), int(c[1]), int(c[0])),thickness,cv2.LINE_AA)
    return cv2.addWeighted( image, image_alpha, mask, alpha, 0 )