import numpy as np
import pandas as pd

# # sub=np.load('X_submission.npy')
# # df=pd.DataFrame(sub, columns=[str(unichr(i)) for i in range(26)])
# # df.to_csv('tot_rec_sub.csv')
# # df.z=df.u +df.v + df.w + df.x + df.y


# df=pd.read_csv('num_check_2.csv')
# # df.y=df.t +df.u + df.v + df.w + df.x
# ary=np.array(df.values)
# np.save('X_submission_1.npy', ary)


train=np.load('X_training.npy')
print train.shape
df=pd.DataFrame(train, columns=[str(unichr(i)) for i in range(26)])
df.to_csv('train_tmp_2.csv')