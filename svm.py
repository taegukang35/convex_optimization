import numpy as np
import matplotlib.pyplot as plt

# SVM에 사용할 데이터 만들기: (7,0), (0,7)에 몰리도록 생성
n = 100
r1,r2 = np.random.randn(n//2,1),np.random.randn(n//2,1)+7
s1,s2 = np.ones((n//2,1)),-np.ones((n//2,1))
r = np.hstack((np.vstack((r1,r2)),np.vstack((r2,r1))))
S = np.vstack((s1,s2))
points = np.hstack((S,r))
X,Y = np.array(points[:,1]),np.array(points[:,2])
X,Y = X.reshape(n,1), Y.reshape(n,1)

# 초기값 세팅
mu = np.ones((n,1))
w = np.vstack((-1,1,0,mu)) # w=[a,b,c,mu]^T
t = 1

# Dual Interior Point Method; Newton Method로 zero-finding
for k in range(50):
    # a,b,c,mu 를 업데이트 해야함
    t = 0.9*t
    a,b,c,mu = w[0],w[1],w[2],w[3:]
    print(a,b,c)

    # R(w) 구하기
    dg_dx = np.hstack((-S*X,-S*Y,S))
    R1 = np.vstack((a,b,0))+np.transpose(dg_dx)@mu
    # g_i = si(c-ax-by)+1
    g = c-np.array((a,b)).T@np.vstack((np.transpose(X),np.transpose(Y)))
    g = S*np.transpose(g)+1
    R2 = mu*g+t*np.ones((n,1))
    R = np.vstack((R1,R2))

    # dR_dw 구하기
    dR_dw1 = np.array([[1,0,0],[0,1,0],[0,0,0]])
    dR_dw2 = np.transpose(dg_dx)
    dR_dw3 = np.diag(np.squeeze(mu))@dg_dx
    dR_dw4 = np.diag(np.squeeze(g))
    dR_dw = np.block([[dR_dw1,dR_dw2],[dR_dw3,dR_dw4]])

    # Newton method 로 w 업데이트
    w = w-np.linalg.inv(dR_dw)@R

    # 결과 plot 하기
    xmin, xmax, ymin, ymax = -5, 10, -5, 10
    fig = plt.figure()
    plt.grid()
    plt.scatter(X, Y)
    plt.axis([xmin, xmax, ymin, ymax])
    xx = np.linspace(xmin, xmax, 100)
    plt.plot(xx, np.squeeze(-(a / b) * xx) + (c / b),'r')
    plt.plot(xx, np.squeeze(-(a / b) * xx) + (c + 1 / b),'b')
    plt.plot(xx, np.squeeze(-(a / b) * xx) + (c - 1 / b),'b')
    fig.canvas.draw();plt.pause(0.5);fig.canvas.flush_events()
