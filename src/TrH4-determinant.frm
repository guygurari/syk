Functions sigma,A,op,signDeltaOp,sign,myd;
Indices 
    tt1,tt2,tt3,tt4,tt5,tt6,tt7,
    tt1m1,tt2m1,tt3m1,tt4m1, 
    tt1m2,tt2m2,tt3m2,tt4m2, 
    tt1m3,tt2m3,tt3m3,tt4m3, 
    tt1m4,tt2m4,tt3m4,tt4m4, 
    tt1m5,tt2m5,tt3m5,tt4m5, 
    tt1m6,tt2m6,tt3m6,tt4m6, 
    t1,t2,t3,t4,t5,t6,t7 t;

*
* We have to first use different external \tilde{t}_i indices, otherwise
* they get contracted in the delta functions (form assumes they are summed
* over). Then after all the contractions are done we replace ttimj by tti.
* 

local [tr_op1] = op(tt1,tt2,tt3,tt4,t1,t1);
local [tr_op2] = op(tt1,tt2,tt3,tt4,t1,t2)*op(tt1m1,tt2m1,tt3m1,tt4m1,t2,t1);
local [tr_op3] =
    op(tt1,tt2,tt3,tt4,t1,t2)*
    op(tt1m1,tt2m1,tt3m1,tt4m1,t2,t3)*
    op(tt1m2,tt2m2,tt3m2,tt4m2,t3,t1);
local [tr_op4] =
    op(tt1,tt2,tt3,tt4,t1,t2)*
    op(tt1m1,tt2m1,tt3m1,tt4m1,t2,t3)*
    op(tt1m2,tt2m2,tt3m2,tt4m2,t3,t4)*
    op(tt1m3,tt2m3,tt3m3,tt4m3,t4,t1);
local [tr_op5] =
    op(tt1,tt2,tt3,tt4,t1,t2)*
    op(tt1m1,tt2m1,tt3m1,tt4m1,t2,t3)*
    op(tt1m2,tt2m2,tt3m2,tt4m2,t3,t4)*
    op(tt1m3,tt2m3,tt3m3,tt4m3,t4,t5)*
    op(tt1m4,tt2m4,tt3m4,tt4m4,t5,t1);
local [tr_op6] =
    op(tt1,tt2,tt3,tt4,t1,t2)*
    op(tt1m1,tt2m1,tt3m1,tt4m1,t2,t3)*
    op(tt1m2,tt2m2,tt3m2,tt4m2,t3,t4)*
    op(tt1m3,tt2m3,tt3m3,tt4m3,t4,t5)*
    op(tt1m4,tt2m4,tt3m4,tt4m4,t5,t6)*
    op(tt1m5,tt2m5,tt3m5,tt4m5,t6,t1);
local [tr_op7] =
    op(tt1,tt2,tt3,tt4,t1,t2)*
    op(tt1m1,tt2m1,tt3m1,tt4m1,t2,t3)*
    op(tt1m2,tt2m2,tt3m2,tt4m2,t3,t4)*
    op(tt1m3,tt2m3,tt3m3,tt4m3,t4,t5)*
    op(tt1m4,tt2m4,tt3m4,tt4m4,t5,t6)*
    op(tt1m5,tt2m5,tt3m5,tt4m5,t6,t7)*
    op(tt1m6,tt2m6,tt3m6,tt4m6,t7,t1);

repeat;

contract;

id op(tt1?,tt2?,tt3?,tt4?,t1?,t2?) = (A(tt1,tt2,t1,t2) + A(tt3,tt4,t1,t2));

id A(tt1?,tt2?,t1?,t2?) = 
    signDeltaOp(tt1,tt2,t1,t2) - signDeltaOp(tt2,tt1,t1,t2);
id signDeltaOp(tt1?,tt2?,t1?,t2?) = sign(t1,tt1) * d_(t2,tt2);

endrepeat;

repeat;

* Replace placeholder ttimj -> tti
id sign(tt1m1,t?) = sign(tt1,t); id sign(tt2m1,t?) = sign(tt2,t);
id sign(tt3m1,t?) = sign(tt3,t); id sign(tt4m1,t?) = sign(tt4,t);
id sign(t?,tt1m1) = sign(t,tt1); id sign(t?,tt2m1) = sign(t,tt2);
id sign(t?,tt3m1) = sign(t,tt3); id sign(t?,tt4m1) = sign(t,tt4);

id sign(tt1m2,t?) = sign(tt1,t); id sign(tt2m2,t?) = sign(tt2,t);
id sign(tt3m2,t?) = sign(tt3,t); id sign(tt4m2,t?) = sign(tt4,t);
id sign(t?,tt1m2) = sign(t,tt1); id sign(t?,tt2m2) = sign(t,tt2);
id sign(t?,tt3m2) = sign(t,tt3); id sign(t?,tt4m2) = sign(t,tt4);

id sign(tt1m3,t?) = sign(tt1,t); id sign(tt2m3,t?) = sign(tt2,t);
id sign(tt3m3,t?) = sign(tt3,t); id sign(tt4m3,t?) = sign(tt4,t);
id sign(t?,tt1m3) = sign(t,tt1); id sign(t?,tt2m3) = sign(t,tt2);
id sign(t?,tt3m3) = sign(t,tt3); id sign(t?,tt4m3) = sign(t,tt4);

id sign(tt1m4,t?) = sign(tt1,t); id sign(tt2m4,t?) = sign(tt2,t);
id sign(tt3m4,t?) = sign(tt3,t); id sign(tt4m4,t?) = sign(tt4,t);
id sign(t?,tt1m4) = sign(t,tt1); id sign(t?,tt2m4) = sign(t,tt2);
id sign(t?,tt3m4) = sign(t,tt3); id sign(t?,tt4m4) = sign(t,tt4);

id sign(tt1m5,t?) = sign(tt1,t); id sign(tt2m5,t?) = sign(tt2,t);
id sign(tt3m5,t?) = sign(tt3,t); id sign(tt4m5,t?) = sign(tt4,t);
id sign(t?,tt1m5) = sign(t,tt1); id sign(t?,tt2m5) = sign(t,tt2);
id sign(t?,tt3m5) = sign(t,tt3); id sign(t?,tt4m5) = sign(t,tt4);

id sign(tt1m6,t?) = sign(tt1,t); id sign(tt2m6,t?) = sign(tt2,t);
id sign(tt3m6,t?) = sign(tt3,t); id sign(tt4m6,t?) = sign(tt4,t);
id sign(t?,tt1m6) = sign(t,tt1); id sign(t?,tt2m6) = sign(t,tt2);
id sign(t?,tt3m6) = sign(t,tt3); id sign(t?,tt4m6) = sign(t,tt4);

* Basic sign properties
id sign(t1?,t1?) = 0;
id sign(t1?,t2?)*sign(t1?,t2?) = 1;

* Put signs in canonical order
id sign(tt2,tt1) = -sign(tt1,tt2);
id sign(tt3,tt1) = -sign(tt1,tt3);
id sign(tt4,tt1) = -sign(tt1,tt4);
id sign(tt3,tt2) = -sign(tt2,tt3);
id sign(tt4,tt2) = -sign(tt2,tt4);
id sign(tt4,tt3) = -sign(tt3,tt4);

* Our choice of time ordering
id sign(tt1,tt2) = -1;
id sign(tt1,tt3) = -1;
id sign(tt1,tt4) = -1;
id sign(tt3,tt4) = -1;

id sign(tt2,tt4)*sign(tt2,tt3) = sign(tt2,tt3)*sign(tt2,tt4);

* Nice little lemma, assuming tt3 < tt4
id sign(t?,tt3)*sign(t?,tt4) = 1 - sign(t,tt3) + sign(t,tt4);

endrepeat;

Print;
.end

