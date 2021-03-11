import numpy as np
 
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    a = np.loadtxt('data_1000/teeth_1000/model_1.txt')
    b = a.shape[0]
    a = a[np.argsort(a[:,6])]
    a = a[:-12, :]
    n = a.shape[0]
    print(b)
    print(n)
    amax = a.max(axis=0)[6]
    amin = a.min(axis=0)[6]
    print("max" + str(amax))
    print("min" + str(amin))
    
    for b in a:
        b[6] = (b[6] - amin) / (amax - amin)
    a = np.delete(a, [3,4,5], axis=1)
    c = np.empty(shape = (n , 3))
    a = np.hstack((a , c))
    # for b in a:
    #     b[4] = 255 - 255 * b[3]
    #     b[5] = 0
    #     b[6] = 255 * b[3]
    for b in a:
         b[4] = 1 - b[3]
         b[5] = 0
         b[6] = b[3]
    a = np.delete(a, 3, axis=1)
    np.savetxt('o3dtest.xyzrgb',a)