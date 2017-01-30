import numpy as np

def default():
    return np.asarray([
        [[-1, 0, 2, 0, -1, 0, -1, 3, -1, 0, 2, 3, 4, 3, 2, 0, -1, 3, -1, 0, -1, 0, 2, 0, -1],
         [0, 0, 1, 0, 0, -2, 2, 3, 2, -2, 2, 1, -1, 1, 2, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
         [0, -1, 3, -1, 0, 0, -1, 3, -1, 0, 0, -1, 3, -1, 0, 0, -1, 3, -1, 0, 0, -1, 3, -1, 0],
         [0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 3, 3, 3, 3, 3, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 2, 1, -1, 1, 2, -2, 2, 3, 2, -2, 0, 0, 1, 0, 0],
         [2, 1, -1, 1, 2, 1, 3, 1, 3, 1, -1, 1, 4, 1, -1, 1, 3, 1, 3, 1, 2, 1, -1, 1, 2]],

        [[1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1],
         [4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 2, 0, 2, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4],
         [0, 0, 0, 0, 0, -1, -1, 1, -1, -1, 1, 1, 2, 1, 1, -1, -1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -2, -2, -2, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 2, 1, 1, 0, 2, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 1, 2, 4, 2, 1, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1, -1, -2, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1, 0, 1, 0, -1, -1, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, 0, -1, -1, -2, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 2, 0, 1, 1, 4, 1, 1, 0, 2, 1, 2, 0, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 2, 0, 2, 4, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 3, 2, 0, 1, 3, 4, 3, 1, 0, 2, 3, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -2, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -2, -3, -2, 0, -1, -3, -4, -3, -1, 0, -2, -3, -2, 0, 0, 0, -1, 0, 0, 0, -1, 1, 3, 4, -1, -1, 1, 2, 3, 1, 1, 1, 1, 1, 3, 2, 1, -1, -1, 4, 3, 1, -1, 0],
         [0, 0, 2, 0, 0, 0, 1, 2, 1, 0, 2, 2, 3, 2, 2, 0, 1, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -2, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 3, 1, 0, 0, 1, 4, 1, 0, 0, 1, 3, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 3, 4, 3, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -2, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 3, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        ]

    ])

def random():
    return np.asarray([

       [np.asarray(np.random.rand(25)),
        np.asarray(np.random.rand(25)),
        np.asarray(np.random.rand(25)),
        np.asarray(np.random.rand(25)),
        np.asarray(np.random.rand(25)),
        np.asarray(np.random.rand(25))],


       [np.asarray(np.random.rand(150)),
        np.asarray(np.random.rand(150)),
        np.asarray(np.random.rand(150)),
        np.asarray(np.random.rand(150)),
        np.asarray(np.random.rand(150)),
        np.asarray(np.random.rand(150))]
    ])
