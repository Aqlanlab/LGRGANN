import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent/"src"))
from lgrgann_wbocimp.reconstruction_mi import mi_weights
from lgrgann_wbocimp.reconstruction_grey_knn import _grc,_grg,_d,knn_predict_scores,iterative_reconstruct_labels
from lgrgann_wbocimp.wbo_opt import parameterize_class_weights,cv_wfi_score
from lgrgann_wbocimp.eval_metrics import weighted_f1_custom,per_class_f1

class TestGRC:
    def test_cat_match(s):assert _grc(1.0,1.0,"categorical")==1.0
    def test_cat_no_match(s):assert _grc(1.0,2.0,"categorical")==0.0
    def test_cont_match(s):assert _grc(1.0,1.0,"continuous")==1.0

class TestGRG:
    def test_identical(s):assert _grg(np.array([1,2,3]),np.array([1,2,3]),np.array([1/3,1/3,1/3]))==1.0
    def test_different(s):assert _grg(np.array([1,2,3]),np.array([4,5,6]),np.array([1/3,1/3,1/3]))==0.0
    def test_partial(s):assert _grg(np.array([1,2,3,4]),np.array([1,2,9,9]),np.array([0.25,0.25,0.25,0.25]))==0.5

class TestWeightedVote:
    def test_dist(s):assert _d(1.0)==0.0;assert _d(0.0)==1.0
    def test_knn_perfect(s):
        Xr,yr=np.array([[1,2,3],[4,5,6]]),np.array(["A","B"])
        sc=knn_predict_scores(Xr,yr,np.array([[1,2,3]]),np.array(["A","B"]),1,np.ones(3)/3)
        assert sc[0,0]==1.0

class TestPriorityConstraint:
    def test_highest(s):
        w=parameterize_class_weights(0,np.array([0.5,0.5]),0.8)
        assert w[0]>=w[1]and w[0]>=w[2]and abs(w.sum()-1.0)<1e-10
    def test_extreme(s):
        w=parameterize_class_weights(0,np.array([0.0,0.0]),1.0)
        assert w[0]==1.0 and w[1]==0.0 and w[2]==0.0

class TestWeightedF1:
    def test_perfect(s):
        yt,yp=np.array([0,0,1,1,2,2]),np.array([0,0,1,1,2,2])
        assert abs(weighted_f1_custom(yt,yp,np.array([1/3,1/3,1/3]),[0,1,2])-1.0)<1e-10

class TestMI:
    def test_sum_one(s):
        X,y=np.random.randn(100,5),np.random.choice(["A","B","C"],100)
        tau=mi_weights(X,y,task="classif",random_state=42)
        assert abs(tau.sum()-1.0)<1e-10 and len(tau)==5

class TestIterative:
    def test_convergence(s):
        np.random.seed(42);n,cls=50,np.array(["A","B","C"])
        X,yt=np.random.randn(n,5),np.random.choice(cls,n)
        lm=np.ones(n,dtype=bool);lm[np.random.choice(n,10,replace=False)]=False
        y=yt.copy();y[~lm]="A";tau=np.ones(5)/5
        yr,phi=iterative_reconstruct_labels(X,y,cls,lm,5,tau,max_iter=20)
        assert phi.shape==(n,3)and np.allclose(phi.sum(axis=1),1.0)

if __name__=="__main__":pytest.main([__file__,"-v"])
