Ѓњ
ъ
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
@
Softplus
features"T
activations"T"
Ttype:
2
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02unknown8е

MDN_size/output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameMDN_size/output_layer/kernel

0MDN_size/output_layer/kernel/Read/ReadVariableOpReadVariableOpMDN_size/output_layer/kernel*
_output_shapes

:*
dtype0

MDN_size/output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameMDN_size/output_layer/bias

.MDN_size/output_layer/bias/Read/ReadVariableOpReadVariableOpMDN_size/output_layer/bias*
_output_shapes
:*
dtype0

MDN_size/alphas/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameMDN_size/alphas/kernel

*MDN_size/alphas/kernel/Read/ReadVariableOpReadVariableOpMDN_size/alphas/kernel*
_output_shapes

:*
dtype0

MDN_size/alphas/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameMDN_size/alphas/bias
y
(MDN_size/alphas/bias/Read/ReadVariableOpReadVariableOpMDN_size/alphas/bias*
_output_shapes
:*
dtype0

MDN_size/distparam1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameMDN_size/distparam1/kernel

.MDN_size/distparam1/kernel/Read/ReadVariableOpReadVariableOpMDN_size/distparam1/kernel*
_output_shapes

:*
dtype0

MDN_size/distparam1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameMDN_size/distparam1/bias

,MDN_size/distparam1/bias/Read/ReadVariableOpReadVariableOpMDN_size/distparam1/bias*
_output_shapes
:*
dtype0

MDN_size/distparam2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameMDN_size/distparam2/kernel

.MDN_size/distparam2/kernel/Read/ReadVariableOpReadVariableOpMDN_size/distparam2/kernel*
_output_shapes

:*
dtype0

MDN_size/distparam2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameMDN_size/distparam2/bias

,MDN_size/distparam2/bias/Read/ReadVariableOpReadVariableOpMDN_size/distparam2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
p

h_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
h_1/kernel
i
h_1/kernel/Read/ReadVariableOpReadVariableOp
h_1/kernel*
_output_shapes

:*
dtype0
h
h_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
h_1/bias
a
h_1/bias/Read/ReadVariableOpReadVariableOph_1/bias*
_output_shapes
:*
dtype0
p

h_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
h_2/kernel
i
h_2/kernel/Read/ReadVariableOpReadVariableOp
h_2/kernel*
_output_shapes

:*
dtype0
h
h_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
h_2/bias
a
h_2/bias/Read/ReadVariableOpReadVariableOph_2/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Ђ
#Adam/MDN_size/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/MDN_size/output_layer/kernel/m

7Adam/MDN_size/output_layer/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/MDN_size/output_layer/kernel/m*
_output_shapes

:*
dtype0

!Adam/MDN_size/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/MDN_size/output_layer/bias/m

5Adam/MDN_size/output_layer/bias/m/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/output_layer/bias/m*
_output_shapes
:*
dtype0

Adam/MDN_size/alphas/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/MDN_size/alphas/kernel/m

1Adam/MDN_size/alphas/kernel/m/Read/ReadVariableOpReadVariableOpAdam/MDN_size/alphas/kernel/m*
_output_shapes

:*
dtype0

Adam/MDN_size/alphas/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/MDN_size/alphas/bias/m

/Adam/MDN_size/alphas/bias/m/Read/ReadVariableOpReadVariableOpAdam/MDN_size/alphas/bias/m*
_output_shapes
:*
dtype0

!Adam/MDN_size/distparam1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/MDN_size/distparam1/kernel/m

5Adam/MDN_size/distparam1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/distparam1/kernel/m*
_output_shapes

:*
dtype0

Adam/MDN_size/distparam1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/MDN_size/distparam1/bias/m

3Adam/MDN_size/distparam1/bias/m/Read/ReadVariableOpReadVariableOpAdam/MDN_size/distparam1/bias/m*
_output_shapes
:*
dtype0

!Adam/MDN_size/distparam2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/MDN_size/distparam2/kernel/m

5Adam/MDN_size/distparam2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/distparam2/kernel/m*
_output_shapes

:*
dtype0

Adam/MDN_size/distparam2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/MDN_size/distparam2/bias/m

3Adam/MDN_size/distparam2/bias/m/Read/ReadVariableOpReadVariableOpAdam/MDN_size/distparam2/bias/m*
_output_shapes
:*
dtype0
~
Adam/h_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/h_1/kernel/m
w
%Adam/h_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h_1/kernel/m*
_output_shapes

:*
dtype0
v
Adam/h_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/h_1/bias/m
o
#Adam/h_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/h_1/bias/m*
_output_shapes
:*
dtype0
~
Adam/h_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/h_2/kernel/m
w
%Adam/h_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h_2/kernel/m*
_output_shapes

:*
dtype0
v
Adam/h_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/h_2/bias/m
o
#Adam/h_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/h_2/bias/m*
_output_shapes
:*
dtype0
Ђ
#Adam/MDN_size/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/MDN_size/output_layer/kernel/v

7Adam/MDN_size/output_layer/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/MDN_size/output_layer/kernel/v*
_output_shapes

:*
dtype0

!Adam/MDN_size/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/MDN_size/output_layer/bias/v

5Adam/MDN_size/output_layer/bias/v/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/output_layer/bias/v*
_output_shapes
:*
dtype0

Adam/MDN_size/alphas/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/MDN_size/alphas/kernel/v

1Adam/MDN_size/alphas/kernel/v/Read/ReadVariableOpReadVariableOpAdam/MDN_size/alphas/kernel/v*
_output_shapes

:*
dtype0

Adam/MDN_size/alphas/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/MDN_size/alphas/bias/v

/Adam/MDN_size/alphas/bias/v/Read/ReadVariableOpReadVariableOpAdam/MDN_size/alphas/bias/v*
_output_shapes
:*
dtype0

!Adam/MDN_size/distparam1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/MDN_size/distparam1/kernel/v

5Adam/MDN_size/distparam1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/distparam1/kernel/v*
_output_shapes

:*
dtype0

Adam/MDN_size/distparam1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/MDN_size/distparam1/bias/v

3Adam/MDN_size/distparam1/bias/v/Read/ReadVariableOpReadVariableOpAdam/MDN_size/distparam1/bias/v*
_output_shapes
:*
dtype0

!Adam/MDN_size/distparam2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/MDN_size/distparam2/kernel/v

5Adam/MDN_size/distparam2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/distparam2/kernel/v*
_output_shapes

:*
dtype0

Adam/MDN_size/distparam2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/MDN_size/distparam2/bias/v

3Adam/MDN_size/distparam2/bias/v/Read/ReadVariableOpReadVariableOpAdam/MDN_size/distparam2/bias/v*
_output_shapes
:*
dtype0
~
Adam/h_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/h_1/kernel/v
w
%Adam/h_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h_1/kernel/v*
_output_shapes

:*
dtype0
v
Adam/h_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/h_1/bias/v
o
#Adam/h_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/h_1/bias/v*
_output_shapes
:*
dtype0
~
Adam/h_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/h_2/kernel/v
w
%Adam/h_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h_2/kernel/v*
_output_shapes

:*
dtype0
v
Adam/h_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/h_2/bias/v
o
#Adam/h_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/h_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ТI
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*§H
valueѓHB№H BщH
У
seqblock
outlayer

alphas

distparam1

distparam2
pvec
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
_
nnmodel
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
R
*trainable_variables
+	variables
,regularization_losses
-	keras_api
А
.iter

/beta_1

0beta_2
	1decay
2learning_ratemmmmmm$m%m3m4m5m 6mЁvЂvЃvЄvЅvІvЇ$vЈ%vЉ3vЊ4vЋ5vЌ6v­
V
30
41
52
63
4
5
6
7
8
9
$10
%11
 
V
30
41
52
63
4
5
6
7
8
9
$10
%11
­
	variables
7non_trainable_variables
8metrics

9layers
:layer_regularization_losses
;layer_metrics
	regularization_losses

trainable_variables
 
К
<layer_with_weights-0
<layer-0
=layer-1
>layer_with_weights-1
>layer-2
?layer-3
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api

30
41
52
63

30
41
52
63
 
­
trainable_variables
	variables
Dnon_trainable_variables
Emetrics
Flayer_regularization_losses
Glayer_metrics
regularization_losses

Hlayers
\Z
VARIABLE_VALUEMDN_size/output_layer/kernel*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEMDN_size/output_layer/bias(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
	variables
Inon_trainable_variables
Jmetrics
Klayer_regularization_losses
Llayer_metrics
regularization_losses

Mlayers
TR
VARIABLE_VALUEMDN_size/alphas/kernel(alphas/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEMDN_size/alphas/bias&alphas/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
	variables
Nnon_trainable_variables
Ometrics
Player_regularization_losses
Qlayer_metrics
regularization_losses

Rlayers
\Z
VARIABLE_VALUEMDN_size/distparam1/kernel,distparam1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEMDN_size/distparam1/bias*distparam1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
 trainable_variables
!	variables
Snon_trainable_variables
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
"regularization_losses

Wlayers
\Z
VARIABLE_VALUEMDN_size/distparam2/kernel,distparam2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEMDN_size/distparam2/bias*distparam2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
­
&trainable_variables
'	variables
Xnon_trainable_variables
Ymetrics
Zlayer_regularization_losses
[layer_metrics
(regularization_losses

\layers
 
 
 
­
*trainable_variables
+	variables
]non_trainable_variables
^metrics
_layer_regularization_losses
`layer_metrics
,regularization_losses

alayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
h_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEh_1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
h_2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEh_2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 

b0
c1
*
0
1
2
3
4
5
 
 
h

3kernel
4bias
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
R
htrainable_variables
i	variables
jregularization_losses
k	keras_api
h

5kernel
6bias
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
R
ptrainable_variables
q	variables
rregularization_losses
s	keras_api

30
41
52
63
 

30
41
52
63
­
@	variables
tnon_trainable_variables
umetrics

vlayers
wlayer_regularization_losses
xlayer_metrics
Aregularization_losses
Btrainable_variables
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ytotal
	zcount
{	variables
|	keras_api
F
	}total
	~count

_fn_kwargs
	variables
	keras_api

30
41

30
41
 
В
dtrainable_variables
e	variables
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
fregularization_losses
layers
 
 
 
В
htrainable_variables
i	variables
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
jregularization_losses
layers

50
61

50
61
 
В
ltrainable_variables
m	variables
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
nregularization_losses
layers
 
 
 
В
ptrainable_variables
q	variables
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
rregularization_losses
layers
 
 

<0
=1
>2
?3
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

{	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

}0
~1

	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
}
VARIABLE_VALUE#Adam/MDN_size/output_layer/kernel/mFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/MDN_size/output_layer/bias/mDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/MDN_size/alphas/kernel/mDalphas/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/MDN_size/alphas/bias/mBalphas/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/MDN_size/distparam1/kernel/mHdistparam1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/MDN_size/distparam1/bias/mFdistparam1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/MDN_size/distparam2/kernel/mHdistparam2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/MDN_size/distparam2/bias/mFdistparam2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/h_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEAdam/h_1/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/h_2/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEAdam/h_2/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE#Adam/MDN_size/output_layer/kernel/vFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/MDN_size/output_layer/bias/vDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/MDN_size/alphas/kernel/vDalphas/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/MDN_size/alphas/bias/vBalphas/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/MDN_size/distparam1/kernel/vHdistparam1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/MDN_size/distparam1/bias/vFdistparam1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/MDN_size/distparam2/kernel/vHdistparam2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/MDN_size/distparam2/bias/vFdistparam2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/h_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEAdam/h_1/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/h_2/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEAdam/h_2/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Ь
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1
h_1/kernelh_1/bias
h_2/kernelh_2/biasMDN_size/output_layer/kernelMDN_size/output_layer/biasMDN_size/alphas/kernelMDN_size/alphas/biasMDN_size/distparam1/kernelMDN_size/distparam1/biasMDN_size/distparam2/kernelMDN_size/distparam2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_77619434
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
л
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0MDN_size/output_layer/kernel/Read/ReadVariableOp.MDN_size/output_layer/bias/Read/ReadVariableOp*MDN_size/alphas/kernel/Read/ReadVariableOp(MDN_size/alphas/bias/Read/ReadVariableOp.MDN_size/distparam1/kernel/Read/ReadVariableOp,MDN_size/distparam1/bias/Read/ReadVariableOp.MDN_size/distparam2/kernel/Read/ReadVariableOp,MDN_size/distparam2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOph_1/kernel/Read/ReadVariableOph_1/bias/Read/ReadVariableOph_2/kernel/Read/ReadVariableOph_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp7Adam/MDN_size/output_layer/kernel/m/Read/ReadVariableOp5Adam/MDN_size/output_layer/bias/m/Read/ReadVariableOp1Adam/MDN_size/alphas/kernel/m/Read/ReadVariableOp/Adam/MDN_size/alphas/bias/m/Read/ReadVariableOp5Adam/MDN_size/distparam1/kernel/m/Read/ReadVariableOp3Adam/MDN_size/distparam1/bias/m/Read/ReadVariableOp5Adam/MDN_size/distparam2/kernel/m/Read/ReadVariableOp3Adam/MDN_size/distparam2/bias/m/Read/ReadVariableOp%Adam/h_1/kernel/m/Read/ReadVariableOp#Adam/h_1/bias/m/Read/ReadVariableOp%Adam/h_2/kernel/m/Read/ReadVariableOp#Adam/h_2/bias/m/Read/ReadVariableOp7Adam/MDN_size/output_layer/kernel/v/Read/ReadVariableOp5Adam/MDN_size/output_layer/bias/v/Read/ReadVariableOp1Adam/MDN_size/alphas/kernel/v/Read/ReadVariableOp/Adam/MDN_size/alphas/bias/v/Read/ReadVariableOp5Adam/MDN_size/distparam1/kernel/v/Read/ReadVariableOp3Adam/MDN_size/distparam1/bias/v/Read/ReadVariableOp5Adam/MDN_size/distparam2/kernel/v/Read/ReadVariableOp3Adam/MDN_size/distparam2/bias/v/Read/ReadVariableOp%Adam/h_1/kernel/v/Read/ReadVariableOp#Adam/h_1/bias/v/Read/ReadVariableOp%Adam/h_2/kernel/v/Read/ReadVariableOp#Adam/h_2/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_77620290
в

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameMDN_size/output_layer/kernelMDN_size/output_layer/biasMDN_size/alphas/kernelMDN_size/alphas/biasMDN_size/distparam1/kernelMDN_size/distparam1/biasMDN_size/distparam2/kernelMDN_size/distparam2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate
h_1/kernelh_1/bias
h_2/kernelh_2/biastotalcounttotal_1count_1#Adam/MDN_size/output_layer/kernel/m!Adam/MDN_size/output_layer/bias/mAdam/MDN_size/alphas/kernel/mAdam/MDN_size/alphas/bias/m!Adam/MDN_size/distparam1/kernel/mAdam/MDN_size/distparam1/bias/m!Adam/MDN_size/distparam2/kernel/mAdam/MDN_size/distparam2/bias/mAdam/h_1/kernel/mAdam/h_1/bias/mAdam/h_2/kernel/mAdam/h_2/bias/m#Adam/MDN_size/output_layer/kernel/v!Adam/MDN_size/output_layer/bias/vAdam/MDN_size/alphas/kernel/vAdam/MDN_size/alphas/bias/v!Adam/MDN_size/distparam1/kernel/vAdam/MDN_size/distparam1/bias/v!Adam/MDN_size/distparam2/kernel/vAdam/MDN_size/distparam2/bias/vAdam/h_1/kernel/vAdam/h_1/bias/vAdam/h_2/kernel/vAdam/h_2/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_77620435ќЩ

љ
H__inference_distparam1_layer_call_and_return_conditional_losses_77619830

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е
д
1__inference_sequential_121_layer_call_fn_77619901

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_sequential_121_layer_call_and_return_conditional_losses_776186362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
h
I__inference_dropout_183_layer_call_and_return_conditional_losses_77618667

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeР
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ё
A__inference_h_1_layer_call_and_return_conditional_losses_77618580

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ,h_1/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ReluЛ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЎ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЩV

#__inference__wrapped_model_77618556
input_1U
Cmdn_size_seqblock_sequential_121_h_1_matmul_readvariableop_resource:R
Dmdn_size_seqblock_sequential_121_h_1_biasadd_readvariableop_resource:U
Cmdn_size_seqblock_sequential_121_h_2_matmul_readvariableop_resource:R
Dmdn_size_seqblock_sequential_121_h_2_biasadd_readvariableop_resource:F
4mdn_size_output_layer_matmul_readvariableop_resource:C
5mdn_size_output_layer_biasadd_readvariableop_resource:@
.mdn_size_alphas_matmul_readvariableop_resource:=
/mdn_size_alphas_biasadd_readvariableop_resource:D
2mdn_size_distparam1_matmul_readvariableop_resource:A
3mdn_size_distparam1_biasadd_readvariableop_resource:D
2mdn_size_distparam2_matmul_readvariableop_resource:A
3mdn_size_distparam2_biasadd_readvariableop_resource:
identityЂ;MDN_size/SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOpЂ:MDN_size/SeqBlock/sequential_121/h_1/MatMul/ReadVariableOpЂ;MDN_size/SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOpЂ:MDN_size/SeqBlock/sequential_121/h_2/MatMul/ReadVariableOpЂ&MDN_size/alphas/BiasAdd/ReadVariableOpЂ%MDN_size/alphas/MatMul/ReadVariableOpЂ*MDN_size/distparam1/BiasAdd/ReadVariableOpЂ)MDN_size/distparam1/MatMul/ReadVariableOpЂ*MDN_size/distparam2/BiasAdd/ReadVariableOpЂ)MDN_size/distparam2/MatMul/ReadVariableOpЂ,MDN_size/output_layer/BiasAdd/ReadVariableOpЂ+MDN_size/output_layer/MatMul/ReadVariableOpќ
:MDN_size/SeqBlock/sequential_121/h_1/MatMul/ReadVariableOpReadVariableOpCmdn_size_seqblock_sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:MDN_size/SeqBlock/sequential_121/h_1/MatMul/ReadVariableOpу
+MDN_size/SeqBlock/sequential_121/h_1/MatMulMatMulinput_1BMDN_size/SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2-
+MDN_size/SeqBlock/sequential_121/h_1/MatMulћ
;MDN_size/SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOpReadVariableOpDmdn_size_seqblock_sequential_121_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;MDN_size/SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp
,MDN_size/SeqBlock/sequential_121/h_1/BiasAddBiasAdd5MDN_size/SeqBlock/sequential_121/h_1/MatMul:product:0CMDN_size/SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,MDN_size/SeqBlock/sequential_121/h_1/BiasAddЧ
)MDN_size/SeqBlock/sequential_121/h_1/ReluRelu5MDN_size/SeqBlock/sequential_121/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2+
)MDN_size/SeqBlock/sequential_121/h_1/Reluх
5MDN_size/SeqBlock/sequential_121/dropout_182/IdentityIdentity7MDN_size/SeqBlock/sequential_121/h_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ27
5MDN_size/SeqBlock/sequential_121/dropout_182/Identityќ
:MDN_size/SeqBlock/sequential_121/h_2/MatMul/ReadVariableOpReadVariableOpCmdn_size_seqblock_sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:MDN_size/SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp
+MDN_size/SeqBlock/sequential_121/h_2/MatMulMatMul>MDN_size/SeqBlock/sequential_121/dropout_182/Identity:output:0BMDN_size/SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2-
+MDN_size/SeqBlock/sequential_121/h_2/MatMulћ
;MDN_size/SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOpReadVariableOpDmdn_size_seqblock_sequential_121_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;MDN_size/SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp
,MDN_size/SeqBlock/sequential_121/h_2/BiasAddBiasAdd5MDN_size/SeqBlock/sequential_121/h_2/MatMul:product:0CMDN_size/SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,MDN_size/SeqBlock/sequential_121/h_2/BiasAddЧ
)MDN_size/SeqBlock/sequential_121/h_2/ReluRelu5MDN_size/SeqBlock/sequential_121/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2+
)MDN_size/SeqBlock/sequential_121/h_2/Reluх
5MDN_size/SeqBlock/sequential_121/dropout_183/IdentityIdentity7MDN_size/SeqBlock/sequential_121/h_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ27
5MDN_size/SeqBlock/sequential_121/dropout_183/IdentityЯ
+MDN_size/output_layer/MatMul/ReadVariableOpReadVariableOp4mdn_size_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+MDN_size/output_layer/MatMul/ReadVariableOpэ
MDN_size/output_layer/MatMulMatMul>MDN_size/SeqBlock/sequential_121/dropout_183/Identity:output:03MDN_size/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/output_layer/MatMulЮ
,MDN_size/output_layer/BiasAdd/ReadVariableOpReadVariableOp5mdn_size_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,MDN_size/output_layer/BiasAdd/ReadVariableOpй
MDN_size/output_layer/BiasAddBiasAdd&MDN_size/output_layer/MatMul:product:04MDN_size/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/output_layer/BiasAdd
MDN_size/output_layer/ReluRelu&MDN_size/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/output_layer/ReluН
%MDN_size/alphas/MatMul/ReadVariableOpReadVariableOp.mdn_size_alphas_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%MDN_size/alphas/MatMul/ReadVariableOpХ
MDN_size/alphas/MatMulMatMul(MDN_size/output_layer/Relu:activations:0-MDN_size/alphas/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/alphas/MatMulМ
&MDN_size/alphas/BiasAdd/ReadVariableOpReadVariableOp/mdn_size_alphas_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&MDN_size/alphas/BiasAdd/ReadVariableOpС
MDN_size/alphas/BiasAddBiasAdd MDN_size/alphas/MatMul:product:0.MDN_size/alphas/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/alphas/BiasAdd
MDN_size/alphas/SoftmaxSoftmax MDN_size/alphas/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/alphas/SoftmaxЩ
)MDN_size/distparam1/MatMul/ReadVariableOpReadVariableOp2mdn_size_distparam1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)MDN_size/distparam1/MatMul/ReadVariableOpб
MDN_size/distparam1/MatMulMatMul(MDN_size/output_layer/Relu:activations:01MDN_size/distparam1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/distparam1/MatMulШ
*MDN_size/distparam1/BiasAdd/ReadVariableOpReadVariableOp3mdn_size_distparam1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*MDN_size/distparam1/BiasAdd/ReadVariableOpб
MDN_size/distparam1/BiasAddBiasAdd$MDN_size/distparam1/MatMul:product:02MDN_size/distparam1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/distparam1/BiasAdd 
MDN_size/distparam1/SoftplusSoftplus$MDN_size/distparam1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/distparam1/SoftplusЩ
)MDN_size/distparam2/MatMul/ReadVariableOpReadVariableOp2mdn_size_distparam2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)MDN_size/distparam2/MatMul/ReadVariableOpб
MDN_size/distparam2/MatMulMatMul(MDN_size/output_layer/Relu:activations:01MDN_size/distparam2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/distparam2/MatMulШ
*MDN_size/distparam2/BiasAdd/ReadVariableOpReadVariableOp3mdn_size_distparam2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*MDN_size/distparam2/BiasAdd/ReadVariableOpб
MDN_size/distparam2/BiasAddBiasAdd$MDN_size/distparam2/MatMul:product:02MDN_size/distparam2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/distparam2/BiasAdd 
MDN_size/distparam2/SoftplusSoftplus$MDN_size/distparam2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/distparam2/Softplusx
MDN_size/pvec/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
MDN_size/pvec/concat/axis
MDN_size/pvec/concatConcatV2!MDN_size/alphas/Softmax:softmax:0*MDN_size/distparam1/Softplus:activations:0*MDN_size/distparam2/Softplus:activations:0"MDN_size/pvec/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
MDN_size/pvec/concatx
IdentityIdentityMDN_size/pvec/concat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЄ
NoOpNoOp<^MDN_size/SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp;^MDN_size/SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp<^MDN_size/SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp;^MDN_size/SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp'^MDN_size/alphas/BiasAdd/ReadVariableOp&^MDN_size/alphas/MatMul/ReadVariableOp+^MDN_size/distparam1/BiasAdd/ReadVariableOp*^MDN_size/distparam1/MatMul/ReadVariableOp+^MDN_size/distparam2/BiasAdd/ReadVariableOp*^MDN_size/distparam2/MatMul/ReadVariableOp-^MDN_size/output_layer/BiasAdd/ReadVariableOp,^MDN_size/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 2z
;MDN_size/SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp;MDN_size/SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp2x
:MDN_size/SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp:MDN_size/SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp2z
;MDN_size/SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp;MDN_size/SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp2x
:MDN_size/SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp:MDN_size/SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp2P
&MDN_size/alphas/BiasAdd/ReadVariableOp&MDN_size/alphas/BiasAdd/ReadVariableOp2N
%MDN_size/alphas/MatMul/ReadVariableOp%MDN_size/alphas/MatMul/ReadVariableOp2X
*MDN_size/distparam1/BiasAdd/ReadVariableOp*MDN_size/distparam1/BiasAdd/ReadVariableOp2V
)MDN_size/distparam1/MatMul/ReadVariableOp)MDN_size/distparam1/MatMul/ReadVariableOp2X
*MDN_size/distparam2/BiasAdd/ReadVariableOp*MDN_size/distparam2/BiasAdd/ReadVariableOp2V
)MDN_size/distparam2/MatMul/ReadVariableOp)MDN_size/distparam2/MatMul/ReadVariableOp2\
,MDN_size/output_layer/BiasAdd/ReadVariableOp,MDN_size/output_layer/BiasAdd/ReadVariableOp2Z
+MDN_size/output_layer/MatMul/ReadVariableOp+MDN_size/output_layer/MatMul/ReadVariableOp:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1

Ё
A__inference_h_1_layer_call_and_return_conditional_losses_77620024

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ,h_1/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ReluЛ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЎ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј
g
.__inference_dropout_183_layer_call_fn_77620093

inputs
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_183_layer_call_and_return_conditional_losses_776186672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы

&__inference_h_2_layer_call_fn_77620066

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_h_2_layer_call_and_return_conditional_losses_776186102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЮH
џ
F__inference_SeqBlock_layer_call_and_return_conditional_losses_77619758

inputsC
1sequential_121_h_1_matmul_readvariableop_resource:@
2sequential_121_h_1_biasadd_readvariableop_resource:C
1sequential_121_h_2_matmul_readvariableop_resource:@
2sequential_121_h_2_biasadd_readvariableop_resource:
identityЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOpЂ)sequential_121/h_1/BiasAdd/ReadVariableOpЂ(sequential_121/h_1/MatMul/ReadVariableOpЂ)sequential_121/h_2/BiasAdd/ReadVariableOpЂ(sequential_121/h_2/MatMul/ReadVariableOpЦ
(sequential_121/h_1/MatMul/ReadVariableOpReadVariableOp1sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential_121/h_1/MatMul/ReadVariableOpЌ
sequential_121/h_1/MatMulMatMulinputs0sequential_121/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/MatMulХ
)sequential_121/h_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_121_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_121/h_1/BiasAdd/ReadVariableOpЭ
sequential_121/h_1/BiasAddBiasAdd#sequential_121/h_1/MatMul:product:01sequential_121/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/BiasAdd
sequential_121/h_1/ReluRelu#sequential_121/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/Relu
(sequential_121/dropout_182/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2*
(sequential_121/dropout_182/dropout/Constу
&sequential_121/dropout_182/dropout/MulMul%sequential_121/h_1/Relu:activations:01sequential_121/dropout_182/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&sequential_121/dropout_182/dropout/MulЉ
(sequential_121/dropout_182/dropout/ShapeShape%sequential_121/h_1/Relu:activations:0*
T0*
_output_shapes
:2*
(sequential_121/dropout_182/dropout/Shape
?sequential_121/dropout_182/dropout/random_uniform/RandomUniformRandomUniform1sequential_121/dropout_182/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg2A
?sequential_121/dropout_182/dropout/random_uniform/RandomUniformЋ
1sequential_121/dropout_182/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>23
1sequential_121/dropout_182/dropout/GreaterEqual/yЊ
/sequential_121/dropout_182/dropout/GreaterEqualGreaterEqualHsequential_121/dropout_182/dropout/random_uniform/RandomUniform:output:0:sequential_121/dropout_182/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ21
/sequential_121/dropout_182/dropout/GreaterEqualа
'sequential_121/dropout_182/dropout/CastCast3sequential_121/dropout_182/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2)
'sequential_121/dropout_182/dropout/Castц
(sequential_121/dropout_182/dropout/Mul_1Mul*sequential_121/dropout_182/dropout/Mul:z:0+sequential_121/dropout_182/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(sequential_121/dropout_182/dropout/Mul_1Ц
(sequential_121/h_2/MatMul/ReadVariableOpReadVariableOp1sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential_121/h_2/MatMul/ReadVariableOpв
sequential_121/h_2/MatMulMatMul,sequential_121/dropout_182/dropout/Mul_1:z:00sequential_121/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/MatMulХ
)sequential_121/h_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_121_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_121/h_2/BiasAdd/ReadVariableOpЭ
sequential_121/h_2/BiasAddBiasAdd#sequential_121/h_2/MatMul:product:01sequential_121/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/BiasAdd
sequential_121/h_2/ReluRelu#sequential_121/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/Relu
(sequential_121/dropout_183/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2*
(sequential_121/dropout_183/dropout/Constу
&sequential_121/dropout_183/dropout/MulMul%sequential_121/h_2/Relu:activations:01sequential_121/dropout_183/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&sequential_121/dropout_183/dropout/MulЉ
(sequential_121/dropout_183/dropout/ShapeShape%sequential_121/h_2/Relu:activations:0*
T0*
_output_shapes
:2*
(sequential_121/dropout_183/dropout/Shape
?sequential_121/dropout_183/dropout/random_uniform/RandomUniformRandomUniform1sequential_121/dropout_183/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg*
seed22A
?sequential_121/dropout_183/dropout/random_uniform/RandomUniformЋ
1sequential_121/dropout_183/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>23
1sequential_121/dropout_183/dropout/GreaterEqual/yЊ
/sequential_121/dropout_183/dropout/GreaterEqualGreaterEqualHsequential_121/dropout_183/dropout/random_uniform/RandomUniform:output:0:sequential_121/dropout_183/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ21
/sequential_121/dropout_183/dropout/GreaterEqualа
'sequential_121/dropout_183/dropout/CastCast3sequential_121/dropout_183/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2)
'sequential_121/dropout_183/dropout/Castц
(sequential_121/dropout_183/dropout/Mul_1Mul*sequential_121/dropout_183/dropout/Mul:z:0+sequential_121/dropout_183/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(sequential_121/dropout_183/dropout/Mul_1Ю
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЮ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mul
IdentityIdentity,sequential_121/dropout_183/dropout/Mul_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityк
NoOpNoOp-^h_1/kernel/Regularizer/Square/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp*^sequential_121/h_1/BiasAdd/ReadVariableOp)^sequential_121/h_1/MatMul/ReadVariableOp*^sequential_121/h_2/BiasAdd/ReadVariableOp)^sequential_121/h_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp2V
)sequential_121/h_1/BiasAdd/ReadVariableOp)sequential_121/h_1/BiasAdd/ReadVariableOp2T
(sequential_121/h_1/MatMul/ReadVariableOp(sequential_121/h_1/MatMul/ReadVariableOp2V
)sequential_121/h_2/BiasAdd/ReadVariableOp)sequential_121/h_2/BiasAdd/ReadVariableOp2T
(sequential_121/h_2/MatMul/ReadVariableOp(sequential_121/h_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ

-__inference_distparam2_layer_call_fn_77619839

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_distparam2_layer_call_and_return_conditional_losses_776189532
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЮH
џ
F__inference_SeqBlock_layer_call_and_return_conditional_losses_77619125

inputsC
1sequential_121_h_1_matmul_readvariableop_resource:@
2sequential_121_h_1_biasadd_readvariableop_resource:C
1sequential_121_h_2_matmul_readvariableop_resource:@
2sequential_121_h_2_biasadd_readvariableop_resource:
identityЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOpЂ)sequential_121/h_1/BiasAdd/ReadVariableOpЂ(sequential_121/h_1/MatMul/ReadVariableOpЂ)sequential_121/h_2/BiasAdd/ReadVariableOpЂ(sequential_121/h_2/MatMul/ReadVariableOpЦ
(sequential_121/h_1/MatMul/ReadVariableOpReadVariableOp1sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential_121/h_1/MatMul/ReadVariableOpЌ
sequential_121/h_1/MatMulMatMulinputs0sequential_121/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/MatMulХ
)sequential_121/h_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_121_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_121/h_1/BiasAdd/ReadVariableOpЭ
sequential_121/h_1/BiasAddBiasAdd#sequential_121/h_1/MatMul:product:01sequential_121/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/BiasAdd
sequential_121/h_1/ReluRelu#sequential_121/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/Relu
(sequential_121/dropout_182/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2*
(sequential_121/dropout_182/dropout/Constу
&sequential_121/dropout_182/dropout/MulMul%sequential_121/h_1/Relu:activations:01sequential_121/dropout_182/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&sequential_121/dropout_182/dropout/MulЉ
(sequential_121/dropout_182/dropout/ShapeShape%sequential_121/h_1/Relu:activations:0*
T0*
_output_shapes
:2*
(sequential_121/dropout_182/dropout/Shape
?sequential_121/dropout_182/dropout/random_uniform/RandomUniformRandomUniform1sequential_121/dropout_182/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg2A
?sequential_121/dropout_182/dropout/random_uniform/RandomUniformЋ
1sequential_121/dropout_182/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>23
1sequential_121/dropout_182/dropout/GreaterEqual/yЊ
/sequential_121/dropout_182/dropout/GreaterEqualGreaterEqualHsequential_121/dropout_182/dropout/random_uniform/RandomUniform:output:0:sequential_121/dropout_182/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ21
/sequential_121/dropout_182/dropout/GreaterEqualа
'sequential_121/dropout_182/dropout/CastCast3sequential_121/dropout_182/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2)
'sequential_121/dropout_182/dropout/Castц
(sequential_121/dropout_182/dropout/Mul_1Mul*sequential_121/dropout_182/dropout/Mul:z:0+sequential_121/dropout_182/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(sequential_121/dropout_182/dropout/Mul_1Ц
(sequential_121/h_2/MatMul/ReadVariableOpReadVariableOp1sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential_121/h_2/MatMul/ReadVariableOpв
sequential_121/h_2/MatMulMatMul,sequential_121/dropout_182/dropout/Mul_1:z:00sequential_121/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/MatMulХ
)sequential_121/h_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_121_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_121/h_2/BiasAdd/ReadVariableOpЭ
sequential_121/h_2/BiasAddBiasAdd#sequential_121/h_2/MatMul:product:01sequential_121/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/BiasAdd
sequential_121/h_2/ReluRelu#sequential_121/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/Relu
(sequential_121/dropout_183/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2*
(sequential_121/dropout_183/dropout/Constу
&sequential_121/dropout_183/dropout/MulMul%sequential_121/h_2/Relu:activations:01sequential_121/dropout_183/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2(
&sequential_121/dropout_183/dropout/MulЉ
(sequential_121/dropout_183/dropout/ShapeShape%sequential_121/h_2/Relu:activations:0*
T0*
_output_shapes
:2*
(sequential_121/dropout_183/dropout/Shape
?sequential_121/dropout_183/dropout/random_uniform/RandomUniformRandomUniform1sequential_121/dropout_183/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg*
seed22A
?sequential_121/dropout_183/dropout/random_uniform/RandomUniformЋ
1sequential_121/dropout_183/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>23
1sequential_121/dropout_183/dropout/GreaterEqual/yЊ
/sequential_121/dropout_183/dropout/GreaterEqualGreaterEqualHsequential_121/dropout_183/dropout/random_uniform/RandomUniform:output:0:sequential_121/dropout_183/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ21
/sequential_121/dropout_183/dropout/GreaterEqualа
'sequential_121/dropout_183/dropout/CastCast3sequential_121/dropout_183/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2)
'sequential_121/dropout_183/dropout/Castц
(sequential_121/dropout_183/dropout/Mul_1Mul*sequential_121/dropout_183/dropout/Mul:z:0+sequential_121/dropout_183/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(sequential_121/dropout_183/dropout/Mul_1Ю
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЮ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mul
IdentityIdentity,sequential_121/dropout_183/dropout/Mul_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityк
NoOpNoOp-^h_1/kernel/Regularizer/Square/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp*^sequential_121/h_1/BiasAdd/ReadVariableOp)^sequential_121/h_1/MatMul/ReadVariableOp*^sequential_121/h_2/BiasAdd/ReadVariableOp)^sequential_121/h_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp2V
)sequential_121/h_1/BiasAdd/ReadVariableOp)sequential_121/h_1/BiasAdd/ReadVariableOp2T
(sequential_121/h_1/MatMul/ReadVariableOp(sequential_121/h_1/MatMul/ReadVariableOp2V
)sequential_121/h_2/BiasAdd/ReadVariableOp)sequential_121/h_2/BiasAdd/ReadVariableOp2T
(sequential_121/h_2/MatMul/ReadVariableOp(sequential_121/h_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ё
A__inference_h_2_layer_call_and_return_conditional_losses_77618610

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ReluЛ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЎ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
Ю
+__inference_SeqBlock_layer_call_fn_77619667

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_776188752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
п
|
B__inference_pvec_layer_call_and_return_conditional_losses_77619865
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
ё

)__inference_alphas_layer_call_fn_77619799

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_alphas_layer_call_and_return_conditional_losses_776189192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е
д
1__inference_sequential_121_layer_call_fn_77619914

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_sequential_121_layer_call_and_return_conditional_losses_776187562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЪР
М
$__inference__traced_restore_77620435
file_prefix?
-assignvariableop_mdn_size_output_layer_kernel:;
-assignvariableop_1_mdn_size_output_layer_bias:;
)assignvariableop_2_mdn_size_alphas_kernel:5
'assignvariableop_3_mdn_size_alphas_bias:?
-assignvariableop_4_mdn_size_distparam1_kernel:9
+assignvariableop_5_mdn_size_distparam1_bias:?
-assignvariableop_6_mdn_size_distparam2_kernel:9
+assignvariableop_7_mdn_size_distparam2_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: 0
assignvariableop_13_h_1_kernel:*
assignvariableop_14_h_1_bias:0
assignvariableop_15_h_2_kernel:*
assignvariableop_16_h_2_bias:#
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: I
7assignvariableop_21_adam_mdn_size_output_layer_kernel_m:C
5assignvariableop_22_adam_mdn_size_output_layer_bias_m:C
1assignvariableop_23_adam_mdn_size_alphas_kernel_m:=
/assignvariableop_24_adam_mdn_size_alphas_bias_m:G
5assignvariableop_25_adam_mdn_size_distparam1_kernel_m:A
3assignvariableop_26_adam_mdn_size_distparam1_bias_m:G
5assignvariableop_27_adam_mdn_size_distparam2_kernel_m:A
3assignvariableop_28_adam_mdn_size_distparam2_bias_m:7
%assignvariableop_29_adam_h_1_kernel_m:1
#assignvariableop_30_adam_h_1_bias_m:7
%assignvariableop_31_adam_h_2_kernel_m:1
#assignvariableop_32_adam_h_2_bias_m:I
7assignvariableop_33_adam_mdn_size_output_layer_kernel_v:C
5assignvariableop_34_adam_mdn_size_output_layer_bias_v:C
1assignvariableop_35_adam_mdn_size_alphas_kernel_v:=
/assignvariableop_36_adam_mdn_size_alphas_bias_v:G
5assignvariableop_37_adam_mdn_size_distparam1_kernel_v:A
3assignvariableop_38_adam_mdn_size_distparam1_bias_v:G
5assignvariableop_39_adam_mdn_size_distparam2_kernel_v:A
3assignvariableop_40_adam_mdn_size_distparam2_bias_v:7
%assignvariableop_41_adam_h_1_kernel_v:1
#assignvariableop_42_adam_h_1_bias_v:7
%assignvariableop_43_adam_h_2_kernel_v:1
#assignvariableop_44_adam_h_2_bias_v:
identity_46ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9ј
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*
valueњBї.B*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB(alphas/kernel/.ATTRIBUTES/VARIABLE_VALUEB&alphas/bias/.ATTRIBUTES/VARIABLE_VALUEB,distparam1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*distparam1/bias/.ATTRIBUTES/VARIABLE_VALUEB,distparam2/kernel/.ATTRIBUTES/VARIABLE_VALUEB*distparam2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDalphas/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBalphas/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHdistparam1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFdistparam1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHdistparam2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFdistparam2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDalphas/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBalphas/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHdistparam1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFdistparam1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHdistparam2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFdistparam2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesъ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЌ
AssignVariableOpAssignVariableOp-assignvariableop_mdn_size_output_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1В
AssignVariableOp_1AssignVariableOp-assignvariableop_1_mdn_size_output_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ў
AssignVariableOp_2AssignVariableOp)assignvariableop_2_mdn_size_alphas_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ќ
AssignVariableOp_3AssignVariableOp'assignvariableop_3_mdn_size_alphas_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4В
AssignVariableOp_4AssignVariableOp-assignvariableop_4_mdn_size_distparam1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5А
AssignVariableOp_5AssignVariableOp+assignvariableop_5_mdn_size_distparam1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6В
AssignVariableOp_6AssignVariableOp-assignvariableop_6_mdn_size_distparam2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7А
AssignVariableOp_7AssignVariableOp+assignvariableop_7_mdn_size_distparam2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8Ё
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ѓ
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ї
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11І
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ў
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13І
AssignVariableOp_13AssignVariableOpassignvariableop_13_h_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Є
AssignVariableOp_14AssignVariableOpassignvariableop_14_h_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15І
AssignVariableOp_15AssignVariableOpassignvariableop_15_h_2_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Є
AssignVariableOp_16AssignVariableOpassignvariableop_16_h_2_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ё
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ё
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ѓ
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ѓ
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21П
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_mdn_size_output_layer_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Н
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adam_mdn_size_output_layer_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Й
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_mdn_size_alphas_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24З
AssignVariableOp_24AssignVariableOp/assignvariableop_24_adam_mdn_size_alphas_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Н
AssignVariableOp_25AssignVariableOp5assignvariableop_25_adam_mdn_size_distparam1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Л
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_mdn_size_distparam1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Н
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_mdn_size_distparam2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Л
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_mdn_size_distparam2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29­
AssignVariableOp_29AssignVariableOp%assignvariableop_29_adam_h_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ћ
AssignVariableOp_30AssignVariableOp#assignvariableop_30_adam_h_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31­
AssignVariableOp_31AssignVariableOp%assignvariableop_31_adam_h_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ћ
AssignVariableOp_32AssignVariableOp#assignvariableop_32_adam_h_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33П
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_mdn_size_output_layer_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Н
AssignVariableOp_34AssignVariableOp5assignvariableop_34_adam_mdn_size_output_layer_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Й
AssignVariableOp_35AssignVariableOp1assignvariableop_35_adam_mdn_size_alphas_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36З
AssignVariableOp_36AssignVariableOp/assignvariableop_36_adam_mdn_size_alphas_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Н
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_mdn_size_distparam1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Л
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_mdn_size_distparam1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Н
AssignVariableOp_39AssignVariableOp5assignvariableop_39_adam_mdn_size_distparam2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Л
AssignVariableOp_40AssignVariableOp3assignvariableop_40_adam_mdn_size_distparam2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41­
AssignVariableOp_41AssignVariableOp%assignvariableop_41_adam_h_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ћ
AssignVariableOp_42AssignVariableOp#assignvariableop_42_adam_h_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43­
AssignVariableOp_43AssignVariableOp%assignvariableop_43_adam_h_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ћ
AssignVariableOp_44AssignVariableOp#assignvariableop_44_adam_h_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpМ
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45f
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_46Є
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

Ї
+__inference_MDN_size_layer_call_fn_77619463

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_MDN_size_layer_call_and_return_conditional_losses_776189882
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

љ
H__inference_distparam1_layer_call_and_return_conditional_losses_77618936

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
у@
ъ
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619379
input_1#
seqblock_77619330:
seqblock_77619332:#
seqblock_77619334:
seqblock_77619336:'
output_layer_77619339:#
output_layer_77619341:!
alphas_77619344:
alphas_77619346:%
distparam1_77619349:!
distparam1_77619351:%
distparam2_77619354:!
distparam2_77619356:
identityЂ>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpЂ SeqBlock/StatefulPartitionedCallЂalphas/StatefulPartitionedCallЂ"distparam1/StatefulPartitionedCallЂ"distparam2/StatefulPartitionedCallЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOpЂ$output_layer/StatefulPartitionedCallХ
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinput_1seqblock_77619330seqblock_77619332seqblock_77619334seqblock_77619336*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_776191252"
 SeqBlock/StatefulPartitionedCallб
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_77619339output_layer_77619341*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_776189022&
$output_layer/StatefulPartitionedCallЗ
alphas/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0alphas_77619344alphas_77619346*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_alphas_layer_call_and_return_conditional_losses_776189192 
alphas/StatefulPartitionedCallЫ
"distparam1/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam1_77619349distparam1_77619351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_distparam1_layer_call_and_return_conditional_losses_776189362$
"distparam1/StatefulPartitionedCallЫ
"distparam2/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam2_77619354distparam2_77619356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_distparam2_layer_call_and_return_conditional_losses_776189532$
"distparam2/StatefulPartitionedCallЧ
pvec/PartitionedCallPartitionedCall'alphas/StatefulPartitionedCall:output:0+distparam1/StatefulPartitionedCall:output:0+distparam2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_pvec_layer_call_and_return_conditional_losses_776189672
pvec/PartitionedCallЎ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_77619330*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЎ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_77619334*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mulж
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_77619339*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpн
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/SquareБ
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Constђ
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/SumЅ
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.MDN_size/output_layer/kernel/Regularizer/mul/xє
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЂ
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp!^SeqBlock/StatefulPartitionedCall^alphas/StatefulPartitionedCall#^distparam1/StatefulPartitionedCall#^distparam2/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 2
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2@
alphas/StatefulPartitionedCallalphas/StatefulPartitionedCall2H
"distparam1/StatefulPartitionedCall"distparam1/StatefulPartitionedCall2H
"distparam2/StatefulPartitionedCall"distparam2/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
і

Ѓ
&__inference_signature_wrapper_77619434
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_776185562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ѕ%

L__inference_sequential_121_layer_call_and_return_conditional_losses_77618808
	h_1_input
h_1_77618783:
h_1_77618785:
h_2_77618789:
h_2_77618791:
identityЂh_1/StatefulPartitionedCallЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂh_2/StatefulPartitionedCallЂ,h_2/kernel/Regularizer/Square/ReadVariableOp
h_1/StatefulPartitionedCallStatefulPartitionedCall	h_1_inputh_1_77618783h_1_77618785*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_h_1_layer_call_and_return_conditional_losses_776185802
h_1/StatefulPartitionedCall§
dropout_182/PartitionedCallPartitionedCall$h_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_182_layer_call_and_return_conditional_losses_776185912
dropout_182/PartitionedCall
h_2/StatefulPartitionedCallStatefulPartitionedCall$dropout_182/PartitionedCall:output:0h_2_77618789h_2_77618791*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_h_2_layer_call_and_return_conditional_losses_776186102
h_2/StatefulPartitionedCall§
dropout_183/PartitionedCallPartitionedCall$h_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_183_layer_call_and_return_conditional_losses_776186212
dropout_183/PartitionedCallЉ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_1_77618783*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЉ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_2_77618789*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mul
IdentityIdentity$dropout_183/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityш
NoOpNoOp^h_1/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp^h_2/StatefulPartitionedCall-^h_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2:
h_2/StatefulPartitionedCallh_2/StatefulPartitionedCall2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	h_1_input
Гi
ј
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619560

inputsL
:seqblock_sequential_121_h_1_matmul_readvariableop_resource:I
;seqblock_sequential_121_h_1_biasadd_readvariableop_resource:L
:seqblock_sequential_121_h_2_matmul_readvariableop_resource:I
;seqblock_sequential_121_h_2_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:7
%alphas_matmul_readvariableop_resource:4
&alphas_biasadd_readvariableop_resource:;
)distparam1_matmul_readvariableop_resource:8
*distparam1_biasadd_readvariableop_resource:;
)distparam2_matmul_readvariableop_resource:8
*distparam2_biasadd_readvariableop_resource:
identityЂ>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpЂ2SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOpЂ1SeqBlock/sequential_121/h_1/MatMul/ReadVariableOpЂ2SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOpЂ1SeqBlock/sequential_121/h_2/MatMul/ReadVariableOpЂalphas/BiasAdd/ReadVariableOpЂalphas/MatMul/ReadVariableOpЂ!distparam1/BiasAdd/ReadVariableOpЂ distparam1/MatMul/ReadVariableOpЂ!distparam2/BiasAdd/ReadVariableOpЂ distparam2/MatMul/ReadVariableOpЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOpЂ#output_layer/BiasAdd/ReadVariableOpЂ"output_layer/MatMul/ReadVariableOpс
1SeqBlock/sequential_121/h_1/MatMul/ReadVariableOpReadVariableOp:seqblock_sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype023
1SeqBlock/sequential_121/h_1/MatMul/ReadVariableOpЧ
"SeqBlock/sequential_121/h_1/MatMulMatMulinputs9SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"SeqBlock/sequential_121/h_1/MatMulр
2SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOpReadVariableOp;seqblock_sequential_121_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOpё
#SeqBlock/sequential_121/h_1/BiasAddBiasAdd,SeqBlock/sequential_121/h_1/MatMul:product:0:SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#SeqBlock/sequential_121/h_1/BiasAddЌ
 SeqBlock/sequential_121/h_1/ReluRelu,SeqBlock/sequential_121/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 SeqBlock/sequential_121/h_1/ReluЪ
,SeqBlock/sequential_121/dropout_182/IdentityIdentity.SeqBlock/sequential_121/h_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,SeqBlock/sequential_121/dropout_182/Identityс
1SeqBlock/sequential_121/h_2/MatMul/ReadVariableOpReadVariableOp:seqblock_sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype023
1SeqBlock/sequential_121/h_2/MatMul/ReadVariableOpі
"SeqBlock/sequential_121/h_2/MatMulMatMul5SeqBlock/sequential_121/dropout_182/Identity:output:09SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"SeqBlock/sequential_121/h_2/MatMulр
2SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOpReadVariableOp;seqblock_sequential_121_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOpё
#SeqBlock/sequential_121/h_2/BiasAddBiasAdd,SeqBlock/sequential_121/h_2/MatMul:product:0:SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#SeqBlock/sequential_121/h_2/BiasAddЌ
 SeqBlock/sequential_121/h_2/ReluRelu,SeqBlock/sequential_121/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 SeqBlock/sequential_121/h_2/ReluЪ
,SeqBlock/sequential_121/dropout_183/IdentityIdentity.SeqBlock/sequential_121/h_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,SeqBlock/sequential_121/dropout_183/IdentityД
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"output_layer/MatMul/ReadVariableOpЩ
output_layer/MatMulMatMul5SeqBlock/sequential_121/dropout_183/Identity:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output_layer/MatMulГ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOpЕ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output_layer/BiasAdd
output_layer/ReluReluoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output_layer/ReluЂ
alphas/MatMul/ReadVariableOpReadVariableOp%alphas_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
alphas/MatMul/ReadVariableOpЁ
alphas/MatMulMatMuloutput_layer/Relu:activations:0$alphas/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
alphas/MatMulЁ
alphas/BiasAdd/ReadVariableOpReadVariableOp&alphas_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
alphas/BiasAdd/ReadVariableOp
alphas/BiasAddBiasAddalphas/MatMul:product:0%alphas/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
alphas/BiasAddv
alphas/SoftmaxSoftmaxalphas/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
alphas/SoftmaxЎ
 distparam1/MatMul/ReadVariableOpReadVariableOp)distparam1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 distparam1/MatMul/ReadVariableOp­
distparam1/MatMulMatMuloutput_layer/Relu:activations:0(distparam1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam1/MatMul­
!distparam1/BiasAdd/ReadVariableOpReadVariableOp*distparam1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!distparam1/BiasAdd/ReadVariableOp­
distparam1/BiasAddBiasAdddistparam1/MatMul:product:0)distparam1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam1/BiasAdd
distparam1/SoftplusSoftplusdistparam1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam1/SoftplusЎ
 distparam2/MatMul/ReadVariableOpReadVariableOp)distparam2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 distparam2/MatMul/ReadVariableOp­
distparam2/MatMulMatMuloutput_layer/Relu:activations:0(distparam2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam2/MatMul­
!distparam2/BiasAdd/ReadVariableOpReadVariableOp*distparam2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!distparam2/BiasAdd/ReadVariableOp­
distparam2/BiasAddBiasAdddistparam2/MatMul:product:0)distparam2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam2/BiasAdd
distparam2/SoftplusSoftplusdistparam2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam2/Softplusf
pvec/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
pvec/concat/axisм
pvec/concatConcatV2alphas/Softmax:softmax:0!distparam1/Softplus:activations:0!distparam2/Softplus:activations:0pvec/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
pvec/concatз
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:seqblock_sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulз
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:seqblock_sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mulь
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpн
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/SquareБ
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Constђ
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/SumЅ
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.MDN_size/output_layer/kernel/Regularizer/mul/xє
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulo
IdentityIdentitypvec/concat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityз
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp3^SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp2^SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp3^SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp2^SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp^alphas/BiasAdd/ReadVariableOp^alphas/MatMul/ReadVariableOp"^distparam1/BiasAdd/ReadVariableOp!^distparam1/MatMul/ReadVariableOp"^distparam2/BiasAdd/ReadVariableOp!^distparam2/MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 2
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2h
2SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp2SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp2f
1SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp1SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp2h
2SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp2SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp2f
1SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp1SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp2>
alphas/BiasAdd/ReadVariableOpalphas/BiasAdd/ReadVariableOp2<
alphas/MatMul/ReadVariableOpalphas/MatMul/ReadVariableOp2F
!distparam1/BiasAdd/ReadVariableOp!distparam1/BiasAdd/ReadVariableOp2D
 distparam1/MatMul/ReadVariableOp distparam1/MatMul/ReadVariableOp2F
!distparam2/BiasAdd/ReadVariableOp!distparam2/BiasAdd/ReadVariableOp2D
 distparam2/MatMul/ReadVariableOp distparam2/MatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ

-__inference_distparam1_layer_call_fn_77619819

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_distparam1_layer_call_and_return_conditional_losses_776189362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
Ю
+__inference_SeqBlock_layer_call_fn_77619680

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_776191252
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о
з
1__inference_sequential_121_layer_call_fn_77618780
	h_1_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	h_1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_sequential_121_layer_call_and_return_conditional_losses_776187562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	h_1_input
е
z
B__inference_pvec_layer_call_and_return_conditional_losses_77618967

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
<

L__inference_sequential_121_layer_call_and_return_conditional_losses_77619992

inputs4
"h_1_matmul_readvariableop_resource:1
#h_1_biasadd_readvariableop_resource:4
"h_2_matmul_readvariableop_resource:1
#h_2_biasadd_readvariableop_resource:
identityЂh_1/BiasAdd/ReadVariableOpЂh_1/MatMul/ReadVariableOpЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂh_2/BiasAdd/ReadVariableOpЂh_2/MatMul/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOp
h_1/MatMul/ReadVariableOpReadVariableOp"h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h_1/MatMul/ReadVariableOp

h_1/MatMulMatMulinputs!h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2

h_1/MatMul
h_1/BiasAdd/ReadVariableOpReadVariableOp#h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h_1/BiasAdd/ReadVariableOp
h_1/BiasAddBiasAddh_1/MatMul:product:0"h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
h_1/BiasAddd
h_1/ReluReluh_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

h_1/Relu{
dropout_182/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_182/dropout/ConstЇ
dropout_182/dropout/MulMulh_1/Relu:activations:0"dropout_182/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_182/dropout/Mul|
dropout_182/dropout/ShapeShapeh_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_182/dropout/Shapeф
0dropout_182/dropout/random_uniform/RandomUniformRandomUniform"dropout_182/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg22
0dropout_182/dropout/random_uniform/RandomUniform
"dropout_182/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"dropout_182/dropout/GreaterEqual/yю
 dropout_182/dropout/GreaterEqualGreaterEqual9dropout_182/dropout/random_uniform/RandomUniform:output:0+dropout_182/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 dropout_182/dropout/GreaterEqualЃ
dropout_182/dropout/CastCast$dropout_182/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_182/dropout/CastЊ
dropout_182/dropout/Mul_1Muldropout_182/dropout/Mul:z:0dropout_182/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_182/dropout/Mul_1
h_2/MatMul/ReadVariableOpReadVariableOp"h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h_2/MatMul/ReadVariableOp

h_2/MatMulMatMuldropout_182/dropout/Mul_1:z:0!h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2

h_2/MatMul
h_2/BiasAdd/ReadVariableOpReadVariableOp#h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h_2/BiasAdd/ReadVariableOp
h_2/BiasAddBiasAddh_2/MatMul:product:0"h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
h_2/BiasAddd
h_2/ReluReluh_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

h_2/Relu{
dropout_183/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout_183/dropout/ConstЇ
dropout_183/dropout/MulMulh_2/Relu:activations:0"dropout_183/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_183/dropout/Mul|
dropout_183/dropout/ShapeShapeh_2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_183/dropout/Shapeё
0dropout_183/dropout/random_uniform/RandomUniformRandomUniform"dropout_183/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg*
seed222
0dropout_183/dropout/random_uniform/RandomUniform
"dropout_183/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"dropout_183/dropout/GreaterEqual/yю
 dropout_183/dropout/GreaterEqualGreaterEqual9dropout_183/dropout/random_uniform/RandomUniform:output:0+dropout_183/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 dropout_183/dropout/GreaterEqualЃ
dropout_183/dropout/CastCast$dropout_183/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout_183/dropout/CastЊ
dropout_183/dropout/Mul_1Muldropout_183/dropout/Mul:z:0dropout_183/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_183/dropout/Mul_1П
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulП
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mulx
IdentityIdentitydropout_183/dropout/Mul_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^h_1/BiasAdd/ReadVariableOp^h_1/MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp^h_2/BiasAdd/ReadVariableOp^h_2/MatMul/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 28
h_1/BiasAdd/ReadVariableOph_1/BiasAdd/ReadVariableOp26
h_1/MatMul/ReadVariableOph_1/MatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp28
h_2/BiasAdd/ReadVariableOph_2/BiasAdd/ReadVariableOp26
h_2/MatMul/ReadVariableOph_2/MatMul/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М(
Ь
L__inference_sequential_121_layer_call_and_return_conditional_losses_77618836
	h_1_input
h_1_77618811:
h_1_77618813:
h_2_77618817:
h_2_77618819:
identityЂ#dropout_182/StatefulPartitionedCallЂ#dropout_183/StatefulPartitionedCallЂh_1/StatefulPartitionedCallЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂh_2/StatefulPartitionedCallЂ,h_2/kernel/Regularizer/Square/ReadVariableOp
h_1/StatefulPartitionedCallStatefulPartitionedCall	h_1_inputh_1_77618811h_1_77618813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_h_1_layer_call_and_return_conditional_losses_776185802
h_1/StatefulPartitionedCall
#dropout_182/StatefulPartitionedCallStatefulPartitionedCall$h_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_182_layer_call_and_return_conditional_losses_776187002%
#dropout_182/StatefulPartitionedCallЇ
h_2/StatefulPartitionedCallStatefulPartitionedCall,dropout_182/StatefulPartitionedCall:output:0h_2_77618817h_2_77618819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_h_2_layer_call_and_return_conditional_losses_776186102
h_2/StatefulPartitionedCallЛ
#dropout_183/StatefulPartitionedCallStatefulPartitionedCall$h_2/StatefulPartitionedCall:output:0$^dropout_182/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_183_layer_call_and_return_conditional_losses_776186672%
#dropout_183/StatefulPartitionedCallЉ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_1_77618811*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЉ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_2_77618817*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mul
IdentityIdentity,dropout_183/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityД
NoOpNoOp$^dropout_182/StatefulPartitionedCall$^dropout_183/StatefulPartitionedCall^h_1/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp^h_2/StatefulPartitionedCall-^h_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 2J
#dropout_182/StatefulPartitionedCall#dropout_182/StatefulPartitionedCall2J
#dropout_183/StatefulPartitionedCall#dropout_183/StatefulPartitionedCall2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2:
h_2/StatefulPartitionedCallh_2/StatefulPartitionedCall2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	h_1_input
Л
h
I__inference_dropout_183_layer_call_and_return_conditional_losses_77620110

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeР
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
g
I__inference_dropout_183_layer_call_and_return_conditional_losses_77618621

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
g
I__inference_dropout_182_layer_call_and_return_conditional_losses_77618591

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

љ
H__inference_distparam2_layer_call_and_return_conditional_losses_77618953

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј
g
.__inference_dropout_182_layer_call_fn_77620034

inputs
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_182_layer_call_and_return_conditional_losses_776187002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р@
щ
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619219

inputs#
seqblock_77619170:
seqblock_77619172:#
seqblock_77619174:
seqblock_77619176:'
output_layer_77619179:#
output_layer_77619181:!
alphas_77619184:
alphas_77619186:%
distparam1_77619189:!
distparam1_77619191:%
distparam2_77619194:!
distparam2_77619196:
identityЂ>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpЂ SeqBlock/StatefulPartitionedCallЂalphas/StatefulPartitionedCallЂ"distparam1/StatefulPartitionedCallЂ"distparam2/StatefulPartitionedCallЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOpЂ$output_layer/StatefulPartitionedCallФ
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinputsseqblock_77619170seqblock_77619172seqblock_77619174seqblock_77619176*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_776191252"
 SeqBlock/StatefulPartitionedCallб
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_77619179output_layer_77619181*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_776189022&
$output_layer/StatefulPartitionedCallЗ
alphas/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0alphas_77619184alphas_77619186*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_alphas_layer_call_and_return_conditional_losses_776189192 
alphas/StatefulPartitionedCallЫ
"distparam1/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam1_77619189distparam1_77619191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_distparam1_layer_call_and_return_conditional_losses_776189362$
"distparam1/StatefulPartitionedCallЫ
"distparam2/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam2_77619194distparam2_77619196*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_distparam2_layer_call_and_return_conditional_losses_776189532$
"distparam2/StatefulPartitionedCallЧ
pvec/PartitionedCallPartitionedCall'alphas/StatefulPartitionedCall:output:0+distparam1/StatefulPartitionedCall:output:0+distparam2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_pvec_layer_call_and_return_conditional_losses_776189672
pvec/PartitionedCallЎ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_77619170*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЎ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_77619174*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mulж
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_77619179*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpн
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/SquareБ
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Constђ
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/SumЅ
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.MDN_size/output_layer/kernel/Regularizer/mul/xє
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЂ
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp!^SeqBlock/StatefulPartitionedCall^alphas/StatefulPartitionedCall#^distparam1/StatefulPartitionedCall#^distparam2/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 2
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2@
alphas/StatefulPartitionedCallalphas/StatefulPartitionedCall2H
"distparam1/StatefulPartitionedCall"distparam1/StatefulPartitionedCall2H
"distparam2/StatefulPartitionedCall"distparam2/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ѕ
D__inference_alphas_layer_call_and_return_conditional_losses_77619810

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
h
I__inference_dropout_182_layer_call_and_return_conditional_losses_77618700

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeР
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

љ
H__inference_distparam2_layer_call_and_return_conditional_losses_77619850

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
g
I__inference_dropout_183_layer_call_and_return_conditional_losses_77620098

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
є
Э
__inference_loss_fn_0_77619876Y
Gmdn_size_output_layer_kernel_regularizer_square_readvariableop_resource:
identityЂ>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpGmdn_size_output_layer_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpн
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/SquareБ
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Constђ
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/SumЅ
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.MDN_size/output_layer/kernel/Regularizer/mul/xє
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulz
IdentityIdentity0MDN_size/output_layer/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp

Ј
+__inference_MDN_size_layer_call_fn_77619275
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_MDN_size_layer_call_and_return_conditional_losses_776192192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1

М
J__inference_output_layer_layer_call_and_return_conditional_losses_77619790

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂ>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Reluп
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpн
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/SquareБ
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Constђ
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/SumЅ
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.MDN_size/output_layer/kernel/Regularizer/mul/xє
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityР
NoOpNoOp^BiasAdd/ReadVariableOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
g
I__inference_dropout_182_layer_call_and_return_conditional_losses_77620039

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

М
J__inference_output_layer_layer_call_and_return_conditional_losses_77618902

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂ>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Reluп
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpн
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/SquareБ
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Constђ
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/SumЅ
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.MDN_size/output_layer/kernel/Regularizer/mul/xє
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityР
NoOpNoOp^BiasAdd/ReadVariableOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Г(
Щ
L__inference_sequential_121_layer_call_and_return_conditional_losses_77618756

inputs
h_1_77618731:
h_1_77618733:
h_2_77618737:
h_2_77618739:
identityЂ#dropout_182/StatefulPartitionedCallЂ#dropout_183/StatefulPartitionedCallЂh_1/StatefulPartitionedCallЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂh_2/StatefulPartitionedCallЂ,h_2/kernel/Regularizer/Square/ReadVariableOp
h_1/StatefulPartitionedCallStatefulPartitionedCallinputsh_1_77618731h_1_77618733*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_h_1_layer_call_and_return_conditional_losses_776185802
h_1/StatefulPartitionedCall
#dropout_182/StatefulPartitionedCallStatefulPartitionedCall$h_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_182_layer_call_and_return_conditional_losses_776187002%
#dropout_182/StatefulPartitionedCallЇ
h_2/StatefulPartitionedCallStatefulPartitionedCall,dropout_182/StatefulPartitionedCall:output:0h_2_77618737h_2_77618739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_h_2_layer_call_and_return_conditional_losses_776186102
h_2/StatefulPartitionedCallЛ
#dropout_183/StatefulPartitionedCallStatefulPartitionedCall$h_2/StatefulPartitionedCall:output:0$^dropout_182/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_183_layer_call_and_return_conditional_losses_776186672%
#dropout_183/StatefulPartitionedCallЉ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_1_77618731*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЉ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_2_77618737*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mul
IdentityIdentity,dropout_183/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityД
NoOpNoOp$^dropout_182/StatefulPartitionedCall$^dropout_183/StatefulPartitionedCall^h_1/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp^h_2/StatefulPartitionedCall-^h_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 2J
#dropout_182/StatefulPartitionedCall#dropout_182/StatefulPartitionedCall2J
#dropout_183/StatefulPartitionedCall#dropout_183/StatefulPartitionedCall2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2:
h_2/StatefulPartitionedCallh_2/StatefulPartitionedCall2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
р@
щ
F__inference_MDN_size_layer_call_and_return_conditional_losses_77618988

inputs#
seqblock_77618876:
seqblock_77618878:#
seqblock_77618880:
seqblock_77618882:'
output_layer_77618903:#
output_layer_77618905:!
alphas_77618920:
alphas_77618922:%
distparam1_77618937:!
distparam1_77618939:%
distparam2_77618954:!
distparam2_77618956:
identityЂ>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpЂ SeqBlock/StatefulPartitionedCallЂalphas/StatefulPartitionedCallЂ"distparam1/StatefulPartitionedCallЂ"distparam2/StatefulPartitionedCallЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOpЂ$output_layer/StatefulPartitionedCallФ
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinputsseqblock_77618876seqblock_77618878seqblock_77618880seqblock_77618882*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_776188752"
 SeqBlock/StatefulPartitionedCallб
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_77618903output_layer_77618905*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_776189022&
$output_layer/StatefulPartitionedCallЗ
alphas/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0alphas_77618920alphas_77618922*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_alphas_layer_call_and_return_conditional_losses_776189192 
alphas/StatefulPartitionedCallЫ
"distparam1/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam1_77618937distparam1_77618939*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_distparam1_layer_call_and_return_conditional_losses_776189362$
"distparam1/StatefulPartitionedCallЫ
"distparam2/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam2_77618954distparam2_77618956*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_distparam2_layer_call_and_return_conditional_losses_776189532$
"distparam2/StatefulPartitionedCallЧ
pvec/PartitionedCallPartitionedCall'alphas/StatefulPartitionedCall:output:0+distparam1/StatefulPartitionedCall:output:0+distparam2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_pvec_layer_call_and_return_conditional_losses_776189672
pvec/PartitionedCallЎ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_77618876*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЎ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_77618880*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mulж
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_77618903*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpн
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/SquareБ
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Constђ
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/SumЅ
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.MDN_size/output_layer/kernel/Regularizer/mul/xє
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЂ
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp!^SeqBlock/StatefulPartitionedCall^alphas/StatefulPartitionedCall#^distparam1/StatefulPartitionedCall#^distparam2/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 2
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2@
alphas/StatefulPartitionedCallalphas/StatefulPartitionedCall2H
"distparam1/StatefulPartitionedCall"distparam1/StatefulPartitionedCall2H
"distparam2/StatefulPartitionedCall"distparam2/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц
J
.__inference_dropout_183_layer_call_fn_77620088

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_183_layer_call_and_return_conditional_losses_776186212
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ј
+__inference_MDN_size_layer_call_fn_77619015
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_MDN_size_layer_call_and_return_conditional_losses_776189882
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
%
§
L__inference_sequential_121_layer_call_and_return_conditional_losses_77618636

inputs
h_1_77618581:
h_1_77618583:
h_2_77618611:
h_2_77618613:
identityЂh_1/StatefulPartitionedCallЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂh_2/StatefulPartitionedCallЂ,h_2/kernel/Regularizer/Square/ReadVariableOp
h_1/StatefulPartitionedCallStatefulPartitionedCallinputsh_1_77618581h_1_77618583*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_h_1_layer_call_and_return_conditional_losses_776185802
h_1/StatefulPartitionedCall§
dropout_182/PartitionedCallPartitionedCall$h_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_182_layer_call_and_return_conditional_losses_776185912
dropout_182/PartitionedCall
h_2/StatefulPartitionedCallStatefulPartitionedCall$dropout_182/PartitionedCall:output:0h_2_77618611h_2_77618613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_h_2_layer_call_and_return_conditional_losses_776186102
h_2/StatefulPartitionedCall§
dropout_183/PartitionedCallPartitionedCall$h_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_183_layer_call_and_return_conditional_losses_776186212
dropout_183/PartitionedCallЉ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_1_77618581*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЉ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_2_77618611*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mul
IdentityIdentity$dropout_183/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityш
NoOpNoOp^h_1/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp^h_2/StatefulPartitionedCall-^h_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2:
h_2/StatefulPartitionedCallh_2/StatefulPartitionedCall2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц
J
.__inference_dropout_182_layer_call_fn_77620029

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dropout_182_layer_call_and_return_conditional_losses_776185912
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ї
+__inference_MDN_size_layer_call_fn_77619492

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_MDN_size_layer_call_and_return_conditional_losses_776192192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
г(

L__inference_sequential_121_layer_call_and_return_conditional_losses_77619946

inputs4
"h_1_matmul_readvariableop_resource:1
#h_1_biasadd_readvariableop_resource:4
"h_2_matmul_readvariableop_resource:1
#h_2_biasadd_readvariableop_resource:
identityЂh_1/BiasAdd/ReadVariableOpЂh_1/MatMul/ReadVariableOpЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂh_2/BiasAdd/ReadVariableOpЂh_2/MatMul/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOp
h_1/MatMul/ReadVariableOpReadVariableOp"h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h_1/MatMul/ReadVariableOp

h_1/MatMulMatMulinputs!h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2

h_1/MatMul
h_1/BiasAdd/ReadVariableOpReadVariableOp#h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h_1/BiasAdd/ReadVariableOp
h_1/BiasAddBiasAddh_1/MatMul:product:0"h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
h_1/BiasAddd
h_1/ReluReluh_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

h_1/Relu
dropout_182/IdentityIdentityh_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_182/Identity
h_2/MatMul/ReadVariableOpReadVariableOp"h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h_2/MatMul/ReadVariableOp

h_2/MatMulMatMuldropout_182/Identity:output:0!h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2

h_2/MatMul
h_2/BiasAdd/ReadVariableOpReadVariableOp#h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h_2/BiasAdd/ReadVariableOp
h_2/BiasAddBiasAddh_2/MatMul:product:0"h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
h_2/BiasAddd
h_2/ReluReluh_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

h_2/Relu
dropout_183/IdentityIdentityh_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout_183/IdentityП
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulП
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mulx
IdentityIdentitydropout_183/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^h_1/BiasAdd/ReadVariableOp^h_1/MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp^h_2/BiasAdd/ReadVariableOp^h_2/MatMul/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 28
h_1/BiasAdd/ReadVariableOph_1/BiasAdd/ReadVariableOp26
h_1/MatMul/ReadVariableOph_1/MatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp28
h_2/BiasAdd/ReadVariableOph_2/BiasAdd/ReadVariableOp26
h_2/MatMul/ReadVariableOph_2/MatMul/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Ё
A__inference_h_2_layer_call_and_return_conditional_losses_77620083

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
ReluЛ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЎ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ\
ю
!__inference__traced_save_77620290
file_prefix;
7savev2_mdn_size_output_layer_kernel_read_readvariableop9
5savev2_mdn_size_output_layer_bias_read_readvariableop5
1savev2_mdn_size_alphas_kernel_read_readvariableop3
/savev2_mdn_size_alphas_bias_read_readvariableop9
5savev2_mdn_size_distparam1_kernel_read_readvariableop7
3savev2_mdn_size_distparam1_bias_read_readvariableop9
5savev2_mdn_size_distparam2_kernel_read_readvariableop7
3savev2_mdn_size_distparam2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop)
%savev2_h_1_kernel_read_readvariableop'
#savev2_h_1_bias_read_readvariableop)
%savev2_h_2_kernel_read_readvariableop'
#savev2_h_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopB
>savev2_adam_mdn_size_output_layer_kernel_m_read_readvariableop@
<savev2_adam_mdn_size_output_layer_bias_m_read_readvariableop<
8savev2_adam_mdn_size_alphas_kernel_m_read_readvariableop:
6savev2_adam_mdn_size_alphas_bias_m_read_readvariableop@
<savev2_adam_mdn_size_distparam1_kernel_m_read_readvariableop>
:savev2_adam_mdn_size_distparam1_bias_m_read_readvariableop@
<savev2_adam_mdn_size_distparam2_kernel_m_read_readvariableop>
:savev2_adam_mdn_size_distparam2_bias_m_read_readvariableop0
,savev2_adam_h_1_kernel_m_read_readvariableop.
*savev2_adam_h_1_bias_m_read_readvariableop0
,savev2_adam_h_2_kernel_m_read_readvariableop.
*savev2_adam_h_2_bias_m_read_readvariableopB
>savev2_adam_mdn_size_output_layer_kernel_v_read_readvariableop@
<savev2_adam_mdn_size_output_layer_bias_v_read_readvariableop<
8savev2_adam_mdn_size_alphas_kernel_v_read_readvariableop:
6savev2_adam_mdn_size_alphas_bias_v_read_readvariableop@
<savev2_adam_mdn_size_distparam1_kernel_v_read_readvariableop>
:savev2_adam_mdn_size_distparam1_bias_v_read_readvariableop@
<savev2_adam_mdn_size_distparam2_kernel_v_read_readvariableop>
:savev2_adam_mdn_size_distparam2_bias_v_read_readvariableop0
,savev2_adam_h_1_kernel_v_read_readvariableop.
*savev2_adam_h_1_bias_v_read_readvariableop0
,savev2_adam_h_2_kernel_v_read_readvariableop.
*savev2_adam_h_2_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameђ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*
valueњBї.B*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB(alphas/kernel/.ATTRIBUTES/VARIABLE_VALUEB&alphas/bias/.ATTRIBUTES/VARIABLE_VALUEB,distparam1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*distparam1/bias/.ATTRIBUTES/VARIABLE_VALUEB,distparam2/kernel/.ATTRIBUTES/VARIABLE_VALUEB*distparam2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDalphas/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBalphas/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHdistparam1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFdistparam1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHdistparam2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFdistparam2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDalphas/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBalphas/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHdistparam1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFdistparam1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHdistparam2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFdistparam2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesф
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesГ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_mdn_size_output_layer_kernel_read_readvariableop5savev2_mdn_size_output_layer_bias_read_readvariableop1savev2_mdn_size_alphas_kernel_read_readvariableop/savev2_mdn_size_alphas_bias_read_readvariableop5savev2_mdn_size_distparam1_kernel_read_readvariableop3savev2_mdn_size_distparam1_bias_read_readvariableop5savev2_mdn_size_distparam2_kernel_read_readvariableop3savev2_mdn_size_distparam2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop%savev2_h_1_kernel_read_readvariableop#savev2_h_1_bias_read_readvariableop%savev2_h_2_kernel_read_readvariableop#savev2_h_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop>savev2_adam_mdn_size_output_layer_kernel_m_read_readvariableop<savev2_adam_mdn_size_output_layer_bias_m_read_readvariableop8savev2_adam_mdn_size_alphas_kernel_m_read_readvariableop6savev2_adam_mdn_size_alphas_bias_m_read_readvariableop<savev2_adam_mdn_size_distparam1_kernel_m_read_readvariableop:savev2_adam_mdn_size_distparam1_bias_m_read_readvariableop<savev2_adam_mdn_size_distparam2_kernel_m_read_readvariableop:savev2_adam_mdn_size_distparam2_bias_m_read_readvariableop,savev2_adam_h_1_kernel_m_read_readvariableop*savev2_adam_h_1_bias_m_read_readvariableop,savev2_adam_h_2_kernel_m_read_readvariableop*savev2_adam_h_2_bias_m_read_readvariableop>savev2_adam_mdn_size_output_layer_kernel_v_read_readvariableop<savev2_adam_mdn_size_output_layer_bias_v_read_readvariableop8savev2_adam_mdn_size_alphas_kernel_v_read_readvariableop6savev2_adam_mdn_size_alphas_bias_v_read_readvariableop<savev2_adam_mdn_size_distparam1_kernel_v_read_readvariableop:savev2_adam_mdn_size_distparam1_bias_v_read_readvariableop<savev2_adam_mdn_size_distparam2_kernel_v_read_readvariableop:savev2_adam_mdn_size_distparam2_bias_v_read_readvariableop,savev2_adam_h_1_kernel_v_read_readvariableop*savev2_adam_h_1_bias_v_read_readvariableop,savev2_adam_h_2_kernel_v_read_readvariableop*savev2_adam_h_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*Ы
_input_shapesЙ
Ж: ::::::::: : : : : ::::: : : : ::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::.

_output_shapes
: 

ј
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619642

inputsL
:seqblock_sequential_121_h_1_matmul_readvariableop_resource:I
;seqblock_sequential_121_h_1_biasadd_readvariableop_resource:L
:seqblock_sequential_121_h_2_matmul_readvariableop_resource:I
;seqblock_sequential_121_h_2_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:7
%alphas_matmul_readvariableop_resource:4
&alphas_biasadd_readvariableop_resource:;
)distparam1_matmul_readvariableop_resource:8
*distparam1_biasadd_readvariableop_resource:;
)distparam2_matmul_readvariableop_resource:8
*distparam2_biasadd_readvariableop_resource:
identityЂ>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpЂ2SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOpЂ1SeqBlock/sequential_121/h_1/MatMul/ReadVariableOpЂ2SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOpЂ1SeqBlock/sequential_121/h_2/MatMul/ReadVariableOpЂalphas/BiasAdd/ReadVariableOpЂalphas/MatMul/ReadVariableOpЂ!distparam1/BiasAdd/ReadVariableOpЂ distparam1/MatMul/ReadVariableOpЂ!distparam2/BiasAdd/ReadVariableOpЂ distparam2/MatMul/ReadVariableOpЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOpЂ#output_layer/BiasAdd/ReadVariableOpЂ"output_layer/MatMul/ReadVariableOpс
1SeqBlock/sequential_121/h_1/MatMul/ReadVariableOpReadVariableOp:seqblock_sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype023
1SeqBlock/sequential_121/h_1/MatMul/ReadVariableOpЧ
"SeqBlock/sequential_121/h_1/MatMulMatMulinputs9SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"SeqBlock/sequential_121/h_1/MatMulр
2SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOpReadVariableOp;seqblock_sequential_121_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOpё
#SeqBlock/sequential_121/h_1/BiasAddBiasAdd,SeqBlock/sequential_121/h_1/MatMul:product:0:SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#SeqBlock/sequential_121/h_1/BiasAddЌ
 SeqBlock/sequential_121/h_1/ReluRelu,SeqBlock/sequential_121/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 SeqBlock/sequential_121/h_1/ReluЋ
1SeqBlock/sequential_121/dropout_182/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?23
1SeqBlock/sequential_121/dropout_182/dropout/Const
/SeqBlock/sequential_121/dropout_182/dropout/MulMul.SeqBlock/sequential_121/h_1/Relu:activations:0:SeqBlock/sequential_121/dropout_182/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ21
/SeqBlock/sequential_121/dropout_182/dropout/MulФ
1SeqBlock/sequential_121/dropout_182/dropout/ShapeShape.SeqBlock/sequential_121/h_1/Relu:activations:0*
T0*
_output_shapes
:23
1SeqBlock/sequential_121/dropout_182/dropout/ShapeЌ
HSeqBlock/sequential_121/dropout_182/dropout/random_uniform/RandomUniformRandomUniform:SeqBlock/sequential_121/dropout_182/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg2J
HSeqBlock/sequential_121/dropout_182/dropout/random_uniform/RandomUniformН
:SeqBlock/sequential_121/dropout_182/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:SeqBlock/sequential_121/dropout_182/dropout/GreaterEqual/yЮ
8SeqBlock/sequential_121/dropout_182/dropout/GreaterEqualGreaterEqualQSeqBlock/sequential_121/dropout_182/dropout/random_uniform/RandomUniform:output:0CSeqBlock/sequential_121/dropout_182/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2:
8SeqBlock/sequential_121/dropout_182/dropout/GreaterEqualы
0SeqBlock/sequential_121/dropout_182/dropout/CastCast<SeqBlock/sequential_121/dropout_182/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
0SeqBlock/sequential_121/dropout_182/dropout/Cast
1SeqBlock/sequential_121/dropout_182/dropout/Mul_1Mul3SeqBlock/sequential_121/dropout_182/dropout/Mul:z:04SeqBlock/sequential_121/dropout_182/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1SeqBlock/sequential_121/dropout_182/dropout/Mul_1с
1SeqBlock/sequential_121/h_2/MatMul/ReadVariableOpReadVariableOp:seqblock_sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype023
1SeqBlock/sequential_121/h_2/MatMul/ReadVariableOpі
"SeqBlock/sequential_121/h_2/MatMulMatMul5SeqBlock/sequential_121/dropout_182/dropout/Mul_1:z:09SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"SeqBlock/sequential_121/h_2/MatMulр
2SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOpReadVariableOp;seqblock_sequential_121_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOpё
#SeqBlock/sequential_121/h_2/BiasAddBiasAdd,SeqBlock/sequential_121/h_2/MatMul:product:0:SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#SeqBlock/sequential_121/h_2/BiasAddЌ
 SeqBlock/sequential_121/h_2/ReluRelu,SeqBlock/sequential_121/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 SeqBlock/sequential_121/h_2/ReluЋ
1SeqBlock/sequential_121/dropout_183/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?23
1SeqBlock/sequential_121/dropout_183/dropout/Const
/SeqBlock/sequential_121/dropout_183/dropout/MulMul.SeqBlock/sequential_121/h_2/Relu:activations:0:SeqBlock/sequential_121/dropout_183/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ21
/SeqBlock/sequential_121/dropout_183/dropout/MulФ
1SeqBlock/sequential_121/dropout_183/dropout/ShapeShape.SeqBlock/sequential_121/h_2/Relu:activations:0*
T0*
_output_shapes
:23
1SeqBlock/sequential_121/dropout_183/dropout/ShapeЙ
HSeqBlock/sequential_121/dropout_183/dropout/random_uniform/RandomUniformRandomUniform:SeqBlock/sequential_121/dropout_183/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg*
seed22J
HSeqBlock/sequential_121/dropout_183/dropout/random_uniform/RandomUniformН
:SeqBlock/sequential_121/dropout_183/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2<
:SeqBlock/sequential_121/dropout_183/dropout/GreaterEqual/yЮ
8SeqBlock/sequential_121/dropout_183/dropout/GreaterEqualGreaterEqualQSeqBlock/sequential_121/dropout_183/dropout/random_uniform/RandomUniform:output:0CSeqBlock/sequential_121/dropout_183/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2:
8SeqBlock/sequential_121/dropout_183/dropout/GreaterEqualы
0SeqBlock/sequential_121/dropout_183/dropout/CastCast<SeqBlock/sequential_121/dropout_183/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ22
0SeqBlock/sequential_121/dropout_183/dropout/Cast
1SeqBlock/sequential_121/dropout_183/dropout/Mul_1Mul3SeqBlock/sequential_121/dropout_183/dropout/Mul:z:04SeqBlock/sequential_121/dropout_183/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1SeqBlock/sequential_121/dropout_183/dropout/Mul_1Д
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"output_layer/MatMul/ReadVariableOpЩ
output_layer/MatMulMatMul5SeqBlock/sequential_121/dropout_183/dropout/Mul_1:z:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output_layer/MatMulГ
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOpЕ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output_layer/BiasAdd
output_layer/ReluReluoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output_layer/ReluЂ
alphas/MatMul/ReadVariableOpReadVariableOp%alphas_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
alphas/MatMul/ReadVariableOpЁ
alphas/MatMulMatMuloutput_layer/Relu:activations:0$alphas/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
alphas/MatMulЁ
alphas/BiasAdd/ReadVariableOpReadVariableOp&alphas_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
alphas/BiasAdd/ReadVariableOp
alphas/BiasAddBiasAddalphas/MatMul:product:0%alphas/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
alphas/BiasAddv
alphas/SoftmaxSoftmaxalphas/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
alphas/SoftmaxЎ
 distparam1/MatMul/ReadVariableOpReadVariableOp)distparam1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 distparam1/MatMul/ReadVariableOp­
distparam1/MatMulMatMuloutput_layer/Relu:activations:0(distparam1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam1/MatMul­
!distparam1/BiasAdd/ReadVariableOpReadVariableOp*distparam1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!distparam1/BiasAdd/ReadVariableOp­
distparam1/BiasAddBiasAdddistparam1/MatMul:product:0)distparam1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam1/BiasAdd
distparam1/SoftplusSoftplusdistparam1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam1/SoftplusЎ
 distparam2/MatMul/ReadVariableOpReadVariableOp)distparam2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 distparam2/MatMul/ReadVariableOp­
distparam2/MatMulMatMuloutput_layer/Relu:activations:0(distparam2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam2/MatMul­
!distparam2/BiasAdd/ReadVariableOpReadVariableOp*distparam2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!distparam2/BiasAdd/ReadVariableOp­
distparam2/BiasAddBiasAdddistparam2/MatMul:product:0)distparam2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam2/BiasAdd
distparam2/SoftplusSoftplusdistparam2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
distparam2/Softplusf
pvec/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
pvec/concat/axisм
pvec/concatConcatV2alphas/Softmax:softmax:0!distparam1/Softplus:activations:0!distparam2/Softplus:activations:0pvec/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ2
pvec/concatз
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:seqblock_sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulз
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:seqblock_sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mulь
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpн
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/SquareБ
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Constђ
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/SumЅ
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.MDN_size/output_layer/kernel/Regularizer/mul/xє
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulo
IdentityIdentitypvec/concat:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityз
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp3^SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp2^SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp3^SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp2^SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp^alphas/BiasAdd/ReadVariableOp^alphas/MatMul/ReadVariableOp"^distparam1/BiasAdd/ReadVariableOp!^distparam1/MatMul/ReadVariableOp"^distparam2/BiasAdd/ReadVariableOp!^distparam2/MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 2
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2h
2SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp2SeqBlock/sequential_121/h_1/BiasAdd/ReadVariableOp2f
1SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp1SeqBlock/sequential_121/h_1/MatMul/ReadVariableOp2h
2SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp2SeqBlock/sequential_121/h_2/BiasAdd/ReadVariableOp2f
1SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp1SeqBlock/sequential_121/h_2/MatMul/ReadVariableOp2>
alphas/BiasAdd/ReadVariableOpalphas/BiasAdd/ReadVariableOp2<
alphas/MatMul/ReadVariableOpalphas/MatMul/ReadVariableOp2F
!distparam1/BiasAdd/ReadVariableOp!distparam1/BiasAdd/ReadVariableOp2D
 distparam1/MatMul/ReadVariableOp distparam1/MatMul/ReadVariableOp2F
!distparam2/BiasAdd/ReadVariableOp!distparam2/BiasAdd/ReadVariableOp2D
 distparam2/MatMul/ReadVariableOp distparam2/MatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы

&__inference_h_1_layer_call_fn_77620007

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_h_1_layer_call_and_return_conditional_losses_776185802
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ/
џ
F__inference_SeqBlock_layer_call_and_return_conditional_losses_77619712

inputsC
1sequential_121_h_1_matmul_readvariableop_resource:@
2sequential_121_h_1_biasadd_readvariableop_resource:C
1sequential_121_h_2_matmul_readvariableop_resource:@
2sequential_121_h_2_biasadd_readvariableop_resource:
identityЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOpЂ)sequential_121/h_1/BiasAdd/ReadVariableOpЂ(sequential_121/h_1/MatMul/ReadVariableOpЂ)sequential_121/h_2/BiasAdd/ReadVariableOpЂ(sequential_121/h_2/MatMul/ReadVariableOpЦ
(sequential_121/h_1/MatMul/ReadVariableOpReadVariableOp1sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential_121/h_1/MatMul/ReadVariableOpЌ
sequential_121/h_1/MatMulMatMulinputs0sequential_121/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/MatMulХ
)sequential_121/h_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_121_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_121/h_1/BiasAdd/ReadVariableOpЭ
sequential_121/h_1/BiasAddBiasAdd#sequential_121/h_1/MatMul:product:01sequential_121/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/BiasAdd
sequential_121/h_1/ReluRelu#sequential_121/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/ReluЏ
#sequential_121/dropout_182/IdentityIdentity%sequential_121/h_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#sequential_121/dropout_182/IdentityЦ
(sequential_121/h_2/MatMul/ReadVariableOpReadVariableOp1sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential_121/h_2/MatMul/ReadVariableOpв
sequential_121/h_2/MatMulMatMul,sequential_121/dropout_182/Identity:output:00sequential_121/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/MatMulХ
)sequential_121/h_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_121_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_121/h_2/BiasAdd/ReadVariableOpЭ
sequential_121/h_2/BiasAddBiasAdd#sequential_121/h_2/MatMul:product:01sequential_121/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/BiasAdd
sequential_121/h_2/ReluRelu#sequential_121/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/ReluЏ
#sequential_121/dropout_183/IdentityIdentity%sequential_121/h_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#sequential_121/dropout_183/IdentityЮ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЮ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mul
IdentityIdentity,sequential_121/dropout_183/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityк
NoOpNoOp-^h_1/kernel/Regularizer/Square/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp*^sequential_121/h_1/BiasAdd/ReadVariableOp)^sequential_121/h_1/MatMul/ReadVariableOp*^sequential_121/h_2/BiasAdd/ReadVariableOp)^sequential_121/h_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp2V
)sequential_121/h_1/BiasAdd/ReadVariableOp)sequential_121/h_1/BiasAdd/ReadVariableOp2T
(sequential_121/h_1/MatMul/ReadVariableOp(sequential_121/h_1/MatMul/ReadVariableOp2V
)sequential_121/h_2/BiasAdd/ReadVariableOp)sequential_121/h_2/BiasAdd/ReadVariableOp2T
(sequential_121/h_2/MatMul/ReadVariableOp(sequential_121/h_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ/
џ
F__inference_SeqBlock_layer_call_and_return_conditional_losses_77618875

inputsC
1sequential_121_h_1_matmul_readvariableop_resource:@
2sequential_121_h_1_biasadd_readvariableop_resource:C
1sequential_121_h_2_matmul_readvariableop_resource:@
2sequential_121_h_2_biasadd_readvariableop_resource:
identityЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOpЂ)sequential_121/h_1/BiasAdd/ReadVariableOpЂ(sequential_121/h_1/MatMul/ReadVariableOpЂ)sequential_121/h_2/BiasAdd/ReadVariableOpЂ(sequential_121/h_2/MatMul/ReadVariableOpЦ
(sequential_121/h_1/MatMul/ReadVariableOpReadVariableOp1sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential_121/h_1/MatMul/ReadVariableOpЌ
sequential_121/h_1/MatMulMatMulinputs0sequential_121/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/MatMulХ
)sequential_121/h_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_121_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_121/h_1/BiasAdd/ReadVariableOpЭ
sequential_121/h_1/BiasAddBiasAdd#sequential_121/h_1/MatMul:product:01sequential_121/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/BiasAdd
sequential_121/h_1/ReluRelu#sequential_121/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_1/ReluЏ
#sequential_121/dropout_182/IdentityIdentity%sequential_121/h_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#sequential_121/dropout_182/IdentityЦ
(sequential_121/h_2/MatMul/ReadVariableOpReadVariableOp1sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential_121/h_2/MatMul/ReadVariableOpв
sequential_121/h_2/MatMulMatMul,sequential_121/dropout_182/Identity:output:00sequential_121/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/MatMulХ
)sequential_121/h_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_121_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_121/h_2/BiasAdd/ReadVariableOpЭ
sequential_121/h_2/BiasAddBiasAdd#sequential_121/h_2/MatMul:product:01sequential_121/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/BiasAdd
sequential_121/h_2/ReluRelu#sequential_121/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_121/h_2/ReluЏ
#sequential_121/dropout_183/IdentityIdentity%sequential_121/h_2/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#sequential_121/dropout_183/IdentityЮ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_121_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЮ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp1sequential_121_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mul
IdentityIdentity,sequential_121/dropout_183/Identity:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityк
NoOpNoOp-^h_1/kernel/Regularizer/Square/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp*^sequential_121/h_1/BiasAdd/ReadVariableOp)^sequential_121/h_1/MatMul/ReadVariableOp*^sequential_121/h_2/BiasAdd/ReadVariableOp)^sequential_121/h_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp2V
)sequential_121/h_1/BiasAdd/ReadVariableOp)sequential_121/h_1/BiasAdd/ReadVariableOp2T
(sequential_121/h_1/MatMul/ReadVariableOp(sequential_121/h_1/MatMul/ReadVariableOp2V
)sequential_121/h_2/BiasAdd/ReadVariableOp)sequential_121/h_2/BiasAdd/ReadVariableOp2T
(sequential_121/h_2/MatMul/ReadVariableOp(sequential_121/h_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о
з
1__inference_sequential_121_layer_call_fn_77618647
	h_1_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	h_1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_sequential_121_layer_call_and_return_conditional_losses_776186362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	h_1_input
М
a
'__inference_pvec_layer_call_fn_77619857
inputs_0
inputs_1
inputs_2
identityи
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_pvec_layer_call_and_return_conditional_losses_776189672
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2

ѕ
D__inference_alphas_layer_call_and_return_conditional_losses_77618919

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
у@
ъ
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619327
input_1#
seqblock_77619278:
seqblock_77619280:#
seqblock_77619282:
seqblock_77619284:'
output_layer_77619287:#
output_layer_77619289:!
alphas_77619292:
alphas_77619294:%
distparam1_77619297:!
distparam1_77619299:%
distparam2_77619302:!
distparam2_77619304:
identityЂ>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpЂ SeqBlock/StatefulPartitionedCallЂalphas/StatefulPartitionedCallЂ"distparam1/StatefulPartitionedCallЂ"distparam2/StatefulPartitionedCallЂ,h_1/kernel/Regularizer/Square/ReadVariableOpЂ,h_2/kernel/Regularizer/Square/ReadVariableOpЂ$output_layer/StatefulPartitionedCallХ
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinput_1seqblock_77619278seqblock_77619280seqblock_77619282seqblock_77619284*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_776188752"
 SeqBlock/StatefulPartitionedCallб
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_77619287output_layer_77619289*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_776189022&
$output_layer/StatefulPartitionedCallЗ
alphas/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0alphas_77619292alphas_77619294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_alphas_layer_call_and_return_conditional_losses_776189192 
alphas/StatefulPartitionedCallЫ
"distparam1/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam1_77619297distparam1_77619299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_distparam1_layer_call_and_return_conditional_losses_776189362$
"distparam1/StatefulPartitionedCallЫ
"distparam2/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam2_77619302distparam2_77619304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_distparam2_layer_call_and_return_conditional_losses_776189532$
"distparam2/StatefulPartitionedCallЧ
pvec/PartitionedCallPartitionedCall'alphas/StatefulPartitionedCall:output:0+distparam1/StatefulPartitionedCall:output:0+distparam2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_pvec_layer_call_and_return_conditional_losses_776189672
pvec/PartitionedCallЎ
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_77619278*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulЎ
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_77619282*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mulж
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_77619287*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpн
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/SquareБ
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Constђ
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/SumЅ
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:20
.MDN_size/output_layer/kernel/Regularizer/mul/xє
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityЂ
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp!^SeqBlock/StatefulPartitionedCall^alphas/StatefulPartitionedCall#^distparam1/StatefulPartitionedCall#^distparam2/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp-^h_2/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 2
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2@
alphas/StatefulPartitionedCallalphas/StatefulPartitionedCall2H
"distparam1/StatefulPartitionedCall"distparam1/StatefulPartitionedCall2H
"distparam2/StatefulPartitionedCall"distparam2/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Л
h
I__inference_dropout_182_layer_call_and_return_conditional_losses_77620051

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeР
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
dtype0*

seedg2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Т
Љ
__inference_loss_fn_2_77620132G
5h_2_kernel_regularizer_square_readvariableop_resource:
identityЂ,h_2/kernel/Regularizer/Square/ReadVariableOpв
,h_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5h_2_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_2/kernel/Regularizer/Square/ReadVariableOpЇ
h_2/kernel/Regularizer/SquareSquare4h_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_2/kernel/Regularizer/Square
h_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_2/kernel/Regularizer/ConstЊ
h_2/kernel/Regularizer/SumSum!h_2/kernel/Regularizer/Square:y:0%h_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/Sum
h_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_2/kernel/Regularizer/mul/xЌ
h_2/kernel/Regularizer/mulMul%h_2/kernel/Regularizer/mul/x:output:0#h_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_2/kernel/Regularizer/mulh
IdentityIdentityh_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity}
NoOpNoOp-^h_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,h_2/kernel/Regularizer/Square/ReadVariableOp,h_2/kernel/Regularizer/Square/ReadVariableOp
§

/__inference_output_layer_layer_call_fn_77619773

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_776189022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Т
Љ
__inference_loss_fn_1_77620121G
5h_1_kernel_regularizer_square_readvariableop_resource:
identityЂ,h_1/kernel/Regularizer/Square/ReadVariableOpв
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5h_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOpЇ
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/ConstЊ
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
h_1/kernel/Regularizer/mul/xЌ
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulh
IdentityIdentityh_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity}
NoOpNoOp-^h_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ћ
serving_default
;
input_10
serving_default_input_1:0џџџџџџџџџ<
output_10
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:рж
Ж
seqblock
outlayer

alphas

distparam1

distparam2
pvec
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
Ў__call__
+Џ&call_and_return_all_conditional_losses
А_default_save_signature"
_tf_keras_model
Д
nnmodel
trainable_variables
	variables
regularization_losses
	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
З__call__
+И&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
*trainable_variables
+	variables
,regularization_losses
-	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
У
.iter

/beta_1

0beta_2
	1decay
2learning_ratemmmmmm$m%m3m4m5m 6mЁvЂvЃvЄvЅvІvЇ$vЈ%vЉ3vЊ4vЋ5vЌ6v­"
	optimizer
v
30
41
52
63
4
5
6
7
8
9
$10
%11"
trackable_list_wrapper
(
Н0"
trackable_list_wrapper
v
30
41
52
63
4
5
6
7
8
9
$10
%11"
trackable_list_wrapper
Ю
	variables
7non_trainable_variables
8metrics

9layers
:layer_regularization_losses
;layer_metrics
	regularization_losses

trainable_variables
Ў__call__
А_default_save_signature
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
-
Оserving_default"
signature_map

<layer_with_weights-0
<layer-0
=layer-1
>layer_with_weights-1
>layer-2
?layer-3
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"
_tf_keras_sequential
<
30
41
52
63"
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
А
trainable_variables
	variables
Dnon_trainable_variables
Emetrics
Flayer_regularization_losses
Glayer_metrics
regularization_losses

Hlayers
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
.:,2MDN_size/output_layer/kernel
(:&2MDN_size/output_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
Н0"
trackable_list_wrapper
А
trainable_variables
	variables
Inon_trainable_variables
Jmetrics
Klayer_regularization_losses
Llayer_metrics
regularization_losses

Mlayers
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
(:&2MDN_size/alphas/kernel
": 2MDN_size/alphas/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
А
trainable_variables
	variables
Nnon_trainable_variables
Ometrics
Player_regularization_losses
Qlayer_metrics
regularization_losses

Rlayers
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
,:*2MDN_size/distparam1/kernel
&:$2MDN_size/distparam1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
А
 trainable_variables
!	variables
Snon_trainable_variables
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
"regularization_losses

Wlayers
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
,:*2MDN_size/distparam2/kernel
&:$2MDN_size/distparam2/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
А
&trainable_variables
'	variables
Xnon_trainable_variables
Ymetrics
Zlayer_regularization_losses
[layer_metrics
(regularization_losses

\layers
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
*trainable_variables
+	variables
]non_trainable_variables
^metrics
_layer_regularization_losses
`layer_metrics
,regularization_losses

alayers
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
:2
h_1/kernel
:2h_1/bias
:2
h_2/kernel
:2h_2/bias
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Н

3kernel
4bias
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
htrainable_variables
i	variables
jregularization_losses
k	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

5kernel
6bias
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"
_tf_keras_layer
<
30
41
52
63"
trackable_list_wrapper
0
Щ0
Ъ1"
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
А
@	variables
tnon_trainable_variables
umetrics

vlayers
wlayer_regularization_losses
xlayer_metrics
Aregularization_losses
Btrainable_variables
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Н0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
N
	ytotal
	zcount
{	variables
|	keras_api"
_tf_keras_metric
`
	}total
	~count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
(
Щ0"
trackable_list_wrapper
Е
dtrainable_variables
e	variables
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
fregularization_losses
layers
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
htrainable_variables
i	variables
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
jregularization_losses
layers
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
(
Ъ0"
trackable_list_wrapper
Е
ltrainable_variables
m	variables
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
nregularization_losses
layers
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
ptrainable_variables
q	variables
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
rregularization_losses
layers
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
y0
z1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
}0
~1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Щ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ъ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
3:12#Adam/MDN_size/output_layer/kernel/m
-:+2!Adam/MDN_size/output_layer/bias/m
-:+2Adam/MDN_size/alphas/kernel/m
':%2Adam/MDN_size/alphas/bias/m
1:/2!Adam/MDN_size/distparam1/kernel/m
+:)2Adam/MDN_size/distparam1/bias/m
1:/2!Adam/MDN_size/distparam2/kernel/m
+:)2Adam/MDN_size/distparam2/bias/m
!:2Adam/h_1/kernel/m
:2Adam/h_1/bias/m
!:2Adam/h_2/kernel/m
:2Adam/h_2/bias/m
3:12#Adam/MDN_size/output_layer/kernel/v
-:+2!Adam/MDN_size/output_layer/bias/v
-:+2Adam/MDN_size/alphas/kernel/v
':%2Adam/MDN_size/alphas/bias/v
1:/2!Adam/MDN_size/distparam1/kernel/v
+:)2Adam/MDN_size/distparam1/bias/v
1:/2!Adam/MDN_size/distparam2/kernel/v
+:)2Adam/MDN_size/distparam2/bias/v
!:2Adam/h_1/kernel/v
:2Adam/h_1/bias/v
!:2Adam/h_2/kernel/v
:2Adam/h_2/bias/v
э2ъ
+__inference_MDN_size_layer_call_fn_77619015
+__inference_MDN_size_layer_call_fn_77619463
+__inference_MDN_size_layer_call_fn_77619492
+__inference_MDN_size_layer_call_fn_77619275Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
й2ж
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619560
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619642
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619327
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619379Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЮBЫ
#__inference__wrapped_model_77618556input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
+__inference_SeqBlock_layer_call_fn_77619667
+__inference_SeqBlock_layer_call_fn_77619680Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Щ2Ц
F__inference_SeqBlock_layer_call_and_return_conditional_losses_77619712
F__inference_SeqBlock_layer_call_and_return_conditional_losses_77619758Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
й2ж
/__inference_output_layer_layer_call_fn_77619773Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
є2ё
J__inference_output_layer_layer_call_and_return_conditional_losses_77619790Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_alphas_layer_call_fn_77619799Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_alphas_layer_call_and_return_conditional_losses_77619810Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_distparam1_layer_call_fn_77619819Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_distparam1_layer_call_and_return_conditional_losses_77619830Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_distparam2_layer_call_fn_77619839Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_distparam2_layer_call_and_return_conditional_losses_77619850Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_pvec_layer_call_fn_77619857Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_pvec_layer_call_and_return_conditional_losses_77619865Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Е2В
__inference_loss_fn_0_77619876
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ЭBЪ
&__inference_signature_wrapper_77619434input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
1__inference_sequential_121_layer_call_fn_77618647
1__inference_sequential_121_layer_call_fn_77619901
1__inference_sequential_121_layer_call_fn_77619914
1__inference_sequential_121_layer_call_fn_77618780Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ў2ћ
L__inference_sequential_121_layer_call_and_return_conditional_losses_77619946
L__inference_sequential_121_layer_call_and_return_conditional_losses_77619992
L__inference_sequential_121_layer_call_and_return_conditional_losses_77618808
L__inference_sequential_121_layer_call_and_return_conditional_losses_77618836Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
а2Э
&__inference_h_1_layer_call_fn_77620007Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_h_1_layer_call_and_return_conditional_losses_77620024Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
.__inference_dropout_182_layer_call_fn_77620029
.__inference_dropout_182_layer_call_fn_77620034Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
а2Э
I__inference_dropout_182_layer_call_and_return_conditional_losses_77620039
I__inference_dropout_182_layer_call_and_return_conditional_losses_77620051Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
а2Э
&__inference_h_2_layer_call_fn_77620066Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_h_2_layer_call_and_return_conditional_losses_77620083Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
.__inference_dropout_183_layer_call_fn_77620088
.__inference_dropout_183_layer_call_fn_77620093Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
а2Э
I__inference_dropout_183_layer_call_and_return_conditional_losses_77620098
I__inference_dropout_183_layer_call_and_return_conditional_losses_77620110Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Е2В
__inference_loss_fn_1_77620121
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Е2В
__inference_loss_fn_2_77620132
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ Е
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619327k3456$%4Ђ1
*Ђ'
!
input_1џџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 Е
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619379k3456$%4Ђ1
*Ђ'
!
input_1џџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 Д
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619560j3456$%3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 Д
F__inference_MDN_size_layer_call_and_return_conditional_losses_77619642j3456$%3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 
+__inference_MDN_size_layer_call_fn_77619015^3456$%4Ђ1
*Ђ'
!
input_1џџџџџџџџџ
p 
Њ "џџџџџџџџџ
+__inference_MDN_size_layer_call_fn_77619275^3456$%4Ђ1
*Ђ'
!
input_1џџџџџџџџџ
p
Њ "џџџџџџџџџ
+__inference_MDN_size_layer_call_fn_77619463]3456$%3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
+__inference_MDN_size_layer_call_fn_77619492]3456$%3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџЌ
F__inference_SeqBlock_layer_call_and_return_conditional_losses_77619712b34563Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 Ќ
F__inference_SeqBlock_layer_call_and_return_conditional_losses_77619758b34563Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 
+__inference_SeqBlock_layer_call_fn_77619667U34563Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
+__inference_SeqBlock_layer_call_fn_77619680U34563Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ
#__inference__wrapped_model_77618556u3456$%0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "3Њ0
.
output_1"
output_1џџџџџџџџџЄ
D__inference_alphas_layer_call_and_return_conditional_losses_77619810\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_alphas_layer_call_fn_77619799O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЈ
H__inference_distparam1_layer_call_and_return_conditional_losses_77619830\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
-__inference_distparam1_layer_call_fn_77619819O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЈ
H__inference_distparam2_layer_call_and_return_conditional_losses_77619850\$%/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
-__inference_distparam2_layer_call_fn_77619839O$%/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЉ
I__inference_dropout_182_layer_call_and_return_conditional_losses_77620039\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 Љ
I__inference_dropout_182_layer_call_and_return_conditional_losses_77620051\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 
.__inference_dropout_182_layer_call_fn_77620029O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
.__inference_dropout_182_layer_call_fn_77620034O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџЉ
I__inference_dropout_183_layer_call_and_return_conditional_losses_77620098\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 Љ
I__inference_dropout_183_layer_call_and_return_conditional_losses_77620110\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 
.__inference_dropout_183_layer_call_fn_77620088O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
.__inference_dropout_183_layer_call_fn_77620093O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџЁ
A__inference_h_1_layer_call_and_return_conditional_losses_77620024\34/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 y
&__inference_h_1_layer_call_fn_77620007O34/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЁ
A__inference_h_2_layer_call_and_return_conditional_losses_77620083\56/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 y
&__inference_h_2_layer_call_fn_77620066O56/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ=
__inference_loss_fn_0_77619876Ђ

Ђ 
Њ " =
__inference_loss_fn_1_776201213Ђ

Ђ 
Њ " =
__inference_loss_fn_2_776201325Ђ

Ђ 
Њ " Њ
J__inference_output_layer_layer_call_and_return_conditional_losses_77619790\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
/__inference_output_layer_layer_call_fn_77619773O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџю
B__inference_pvec_layer_call_and_return_conditional_losses_77619865Ї~Ђ{
tЂq
ol
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
"
inputs/2џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 Ц
'__inference_pvec_layer_call_fn_77619857~Ђ{
tЂq
ol
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
"
inputs/2џџџџџџџџџ
Њ "џџџџџџџџџЙ
L__inference_sequential_121_layer_call_and_return_conditional_losses_77618808i3456:Ђ7
0Ђ-
# 
	h_1_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Й
L__inference_sequential_121_layer_call_and_return_conditional_losses_77618836i3456:Ђ7
0Ђ-
# 
	h_1_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ж
L__inference_sequential_121_layer_call_and_return_conditional_losses_77619946f34567Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ж
L__inference_sequential_121_layer_call_and_return_conditional_losses_77619992f34567Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
1__inference_sequential_121_layer_call_fn_77618647\3456:Ђ7
0Ђ-
# 
	h_1_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
1__inference_sequential_121_layer_call_fn_77618780\3456:Ђ7
0Ђ-
# 
	h_1_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
1__inference_sequential_121_layer_call_fn_77619901Y34567Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
1__inference_sequential_121_layer_call_fn_77619914Y34567Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЋ
&__inference_signature_wrapper_776194343456$%;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"3Њ0
.
output_1"
output_1џџџџџџџџџ