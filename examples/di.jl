using ModelPredictiveControl, ControlSystemsBase

Ts = 4.0
A =  [  0.800737  0.0       0.0  0.0
        0.0       0.606531  0.0  0.0
        0.0       0.0       0.8  0.0
        0.0       0.0       0.0  0.6    ]
Bu = [  0.378599  0.378599
        -0.291167  0.291167
        0.0       0.0
        0.0       0.0                   ]
Bd = [  0; 0; 0.5; 0.5;;                ]
C =  [  1.0  0.0  0.684   0.0
        0.0  1.0  0.0    -0.4736        ]
Dd = [  0.19; -0.148;;                  ]
Du = zeros(2,2)
model = LinModel(ss(A,[Bu Bd],C,[Du Dd],Ts),Ts,i_d=[3])
model = setop!(model, uop=[10,10], yop=[50,30], dop=[5])
