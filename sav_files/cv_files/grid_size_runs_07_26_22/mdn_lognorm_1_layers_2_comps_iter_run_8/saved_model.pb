��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
delete_old_dirsbool(�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring �
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02unknown8��
�
MDN_size/output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameMDN_size/output_layer/kernel
�
0MDN_size/output_layer/kernel/Read/ReadVariableOpReadVariableOpMDN_size/output_layer/kernel*
_output_shapes

:*
dtype0
�
MDN_size/output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameMDN_size/output_layer/bias
�
.MDN_size/output_layer/bias/Read/ReadVariableOpReadVariableOpMDN_size/output_layer/bias*
_output_shapes
:*
dtype0
�
MDN_size/alphas/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameMDN_size/alphas/kernel
�
*MDN_size/alphas/kernel/Read/ReadVariableOpReadVariableOpMDN_size/alphas/kernel*
_output_shapes

:*
dtype0
�
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
�
MDN_size/distparam1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameMDN_size/distparam1/kernel
�
.MDN_size/distparam1/kernel/Read/ReadVariableOpReadVariableOpMDN_size/distparam1/kernel*
_output_shapes

:*
dtype0
�
MDN_size/distparam1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameMDN_size/distparam1/bias
�
,MDN_size/distparam1/bias/Read/ReadVariableOpReadVariableOpMDN_size/distparam1/bias*
_output_shapes
:*
dtype0
�
MDN_size/distparam2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameMDN_size/distparam2/kernel
�
.MDN_size/distparam2/kernel/Read/ReadVariableOpReadVariableOpMDN_size/distparam2/kernel*
_output_shapes

:*
dtype0
�
MDN_size/distparam2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameMDN_size/distparam2/bias
�
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
:*
shared_name
h_1/kernel
i
h_1/kernel/Read/ReadVariableOpReadVariableOp
h_1/kernel*
_output_shapes

:*
dtype0
h
h_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
h_1/bias
a
h_1/bias/Read/ReadVariableOpReadVariableOph_1/bias*
_output_shapes
:*
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
�
#Adam/MDN_size/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/MDN_size/output_layer/kernel/m
�
7Adam/MDN_size/output_layer/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/MDN_size/output_layer/kernel/m*
_output_shapes

:*
dtype0
�
!Adam/MDN_size/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/MDN_size/output_layer/bias/m
�
5Adam/MDN_size/output_layer/bias/m/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/output_layer/bias/m*
_output_shapes
:*
dtype0
�
Adam/MDN_size/alphas/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/MDN_size/alphas/kernel/m
�
1Adam/MDN_size/alphas/kernel/m/Read/ReadVariableOpReadVariableOpAdam/MDN_size/alphas/kernel/m*
_output_shapes

:*
dtype0
�
Adam/MDN_size/alphas/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/MDN_size/alphas/bias/m
�
/Adam/MDN_size/alphas/bias/m/Read/ReadVariableOpReadVariableOpAdam/MDN_size/alphas/bias/m*
_output_shapes
:*
dtype0
�
!Adam/MDN_size/distparam1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/MDN_size/distparam1/kernel/m
�
5Adam/MDN_size/distparam1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/distparam1/kernel/m*
_output_shapes

:*
dtype0
�
Adam/MDN_size/distparam1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/MDN_size/distparam1/bias/m
�
3Adam/MDN_size/distparam1/bias/m/Read/ReadVariableOpReadVariableOpAdam/MDN_size/distparam1/bias/m*
_output_shapes
:*
dtype0
�
!Adam/MDN_size/distparam2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/MDN_size/distparam2/kernel/m
�
5Adam/MDN_size/distparam2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/distparam2/kernel/m*
_output_shapes

:*
dtype0
�
Adam/MDN_size/distparam2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/MDN_size/distparam2/bias/m
�
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
:*"
shared_nameAdam/h_1/kernel/m
w
%Adam/h_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h_1/kernel/m*
_output_shapes

:*
dtype0
v
Adam/h_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/h_1/bias/m
o
#Adam/h_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/h_1/bias/m*
_output_shapes
:*
dtype0
�
#Adam/MDN_size/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/MDN_size/output_layer/kernel/v
�
7Adam/MDN_size/output_layer/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/MDN_size/output_layer/kernel/v*
_output_shapes

:*
dtype0
�
!Adam/MDN_size/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/MDN_size/output_layer/bias/v
�
5Adam/MDN_size/output_layer/bias/v/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/output_layer/bias/v*
_output_shapes
:*
dtype0
�
Adam/MDN_size/alphas/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/MDN_size/alphas/kernel/v
�
1Adam/MDN_size/alphas/kernel/v/Read/ReadVariableOpReadVariableOpAdam/MDN_size/alphas/kernel/v*
_output_shapes

:*
dtype0
�
Adam/MDN_size/alphas/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/MDN_size/alphas/bias/v
�
/Adam/MDN_size/alphas/bias/v/Read/ReadVariableOpReadVariableOpAdam/MDN_size/alphas/bias/v*
_output_shapes
:*
dtype0
�
!Adam/MDN_size/distparam1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/MDN_size/distparam1/kernel/v
�
5Adam/MDN_size/distparam1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/distparam1/kernel/v*
_output_shapes

:*
dtype0
�
Adam/MDN_size/distparam1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/MDN_size/distparam1/bias/v
�
3Adam/MDN_size/distparam1/bias/v/Read/ReadVariableOpReadVariableOpAdam/MDN_size/distparam1/bias/v*
_output_shapes
:*
dtype0
�
!Adam/MDN_size/distparam2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/MDN_size/distparam2/kernel/v
�
5Adam/MDN_size/distparam2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/MDN_size/distparam2/kernel/v*
_output_shapes

:*
dtype0
�
Adam/MDN_size/distparam2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/MDN_size/distparam2/bias/v
�
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
:*"
shared_nameAdam/h_1/kernel/v
w
%Adam/h_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h_1/kernel/v*
_output_shapes

:*
dtype0
v
Adam/h_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/h_1/bias/v
o
#Adam/h_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/h_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�>
value�>B�> B�>
�
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
�
.iter

/beta_1

0beta_2
	1decay
2learning_ratem�m�m�m�m�m�$m�%m�3m�4m�v�v�v�v�v�v�$v�%v�3v�4v�
F
30
41
2
3
4
5
6
7
$8
%9
 
F
30
41
2
3
4
5
6
7
$8
%9
�
	variables
5non_trainable_variables
6metrics

7layers
8layer_regularization_losses
9layer_metrics
	regularization_losses

trainable_variables
 
�
:layer_with_weights-0
:layer-0
;layer-1
<	variables
=regularization_losses
>trainable_variables
?	keras_api

30
41

30
41
 
�
trainable_variables
	variables
@non_trainable_variables
Ametrics
Blayer_regularization_losses
Clayer_metrics
regularization_losses

Dlayers
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
�
trainable_variables
	variables
Enon_trainable_variables
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
regularization_losses

Ilayers
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
�
trainable_variables
	variables
Jnon_trainable_variables
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
regularization_losses

Nlayers
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
�
 trainable_variables
!	variables
Onon_trainable_variables
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
"regularization_losses

Slayers
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
�
&trainable_variables
'	variables
Tnon_trainable_variables
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
(regularization_losses

Xlayers
 
 
 
�
*trainable_variables
+	variables
Ynon_trainable_variables
Zmetrics
[layer_regularization_losses
\layer_metrics
,regularization_losses

]layers
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
 

^0
_1
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
`trainable_variables
a	variables
bregularization_losses
c	keras_api
R
dtrainable_variables
e	variables
fregularization_losses
g	keras_api

30
41
 

30
41
�
<	variables
hnon_trainable_variables
imetrics

jlayers
klayer_regularization_losses
llayer_metrics
=regularization_losses
>trainable_variables
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
	mtotal
	ncount
o	variables
p	keras_api
D
	qtotal
	rcount
s
_fn_kwargs
t	variables
u	keras_api

30
41

30
41
 
�
`trainable_variables
a	variables
vnon_trainable_variables
wmetrics
xlayer_regularization_losses
ylayer_metrics
bregularization_losses

zlayers
 
 
 
�
dtrainable_variables
e	variables
{non_trainable_variables
|metrics
}layer_regularization_losses
~layer_metrics
fregularization_losses

layers
 
 

:0
;1
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

m0
n1

o	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

t	variables
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
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1
h_1/kernelh_1/biasMDN_size/output_layer/kernelMDN_size/output_layer/biasMDN_size/alphas/kernelMDN_size/alphas/biasMDN_size/distparam1/kernelMDN_size/distparam1/biasMDN_size/distparam2/kernelMDN_size/distparam2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_27543537
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0MDN_size/output_layer/kernel/Read/ReadVariableOp.MDN_size/output_layer/bias/Read/ReadVariableOp*MDN_size/alphas/kernel/Read/ReadVariableOp(MDN_size/alphas/bias/Read/ReadVariableOp.MDN_size/distparam1/kernel/Read/ReadVariableOp,MDN_size/distparam1/bias/Read/ReadVariableOp.MDN_size/distparam2/kernel/Read/ReadVariableOp,MDN_size/distparam2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOph_1/kernel/Read/ReadVariableOph_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp7Adam/MDN_size/output_layer/kernel/m/Read/ReadVariableOp5Adam/MDN_size/output_layer/bias/m/Read/ReadVariableOp1Adam/MDN_size/alphas/kernel/m/Read/ReadVariableOp/Adam/MDN_size/alphas/bias/m/Read/ReadVariableOp5Adam/MDN_size/distparam1/kernel/m/Read/ReadVariableOp3Adam/MDN_size/distparam1/bias/m/Read/ReadVariableOp5Adam/MDN_size/distparam2/kernel/m/Read/ReadVariableOp3Adam/MDN_size/distparam2/bias/m/Read/ReadVariableOp%Adam/h_1/kernel/m/Read/ReadVariableOp#Adam/h_1/bias/m/Read/ReadVariableOp7Adam/MDN_size/output_layer/kernel/v/Read/ReadVariableOp5Adam/MDN_size/output_layer/bias/v/Read/ReadVariableOp1Adam/MDN_size/alphas/kernel/v/Read/ReadVariableOp/Adam/MDN_size/alphas/bias/v/Read/ReadVariableOp5Adam/MDN_size/distparam1/kernel/v/Read/ReadVariableOp3Adam/MDN_size/distparam1/bias/v/Read/ReadVariableOp5Adam/MDN_size/distparam2/kernel/v/Read/ReadVariableOp3Adam/MDN_size/distparam2/bias/v/Read/ReadVariableOp%Adam/h_1/kernel/v/Read/ReadVariableOp#Adam/h_1/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
GPU 2J 8� **
f%R#
!__inference__traced_save_27544164
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameMDN_size/output_layer/kernelMDN_size/output_layer/biasMDN_size/alphas/kernelMDN_size/alphas/biasMDN_size/distparam1/kernelMDN_size/distparam1/biasMDN_size/distparam2/kernelMDN_size/distparam2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate
h_1/kernelh_1/biastotalcounttotal_1count_1#Adam/MDN_size/output_layer/kernel/m!Adam/MDN_size/output_layer/bias/mAdam/MDN_size/alphas/kernel/mAdam/MDN_size/alphas/bias/m!Adam/MDN_size/distparam1/kernel/mAdam/MDN_size/distparam1/bias/m!Adam/MDN_size/distparam2/kernel/mAdam/MDN_size/distparam2/bias/mAdam/h_1/kernel/mAdam/h_1/bias/m#Adam/MDN_size/output_layer/kernel/v!Adam/MDN_size/output_layer/bias/vAdam/MDN_size/alphas/kernel/vAdam/MDN_size/alphas/bias/v!Adam/MDN_size/distparam1/kernel/vAdam/MDN_size/distparam1/bias/v!Adam/MDN_size/distparam2/kernel/vAdam/MDN_size/distparam2/bias/vAdam/h_1/kernel/vAdam/h_1/bias/v*3
Tin,
*2(*
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
GPU 2J 8� *-
f(R&
$__inference__traced_restore_27544291��

�
�
)__inference_alphas_layer_call_fn_27543810

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_alphas_layer_call_and_return_conditional_losses_275431132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_SeqBlock_layer_call_fn_27543717

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_275430732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543954

inputs4
"h_1_matmul_readvariableop_resource:1
#h_1_biasadd_readvariableop_resource:
identity��h_1/BiasAdd/ReadVariableOp�h_1/MatMul/ReadVariableOp�,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/MatMul/ReadVariableOpReadVariableOp"h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h_1/MatMul/ReadVariableOp

h_1/MatMulMatMulinputs!h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

h_1/MatMul�
h_1/BiasAdd/ReadVariableOpReadVariableOp#h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h_1/BiasAdd/ReadVariableOp�
h_1/BiasAddBiasAddh_1/MatMul:product:0"h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
h_1/BiasAddd
h_1/ReluReluh_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

h_1/Reluy
dropout_70/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_70/dropout/Const�
dropout_70/dropout/MulMulh_1/Relu:activations:0!dropout_70/dropout/Const:output:0*
T0*'
_output_shapes
:���������2
dropout_70/dropout/Mulz
dropout_70/dropout/ShapeShapeh_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_70/dropout/Shape�
/dropout_70/dropout/random_uniform/RandomUniformRandomUniform!dropout_70/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed�21
/dropout_70/dropout/random_uniform/RandomUniform�
!dropout_70/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2#
!dropout_70/dropout/GreaterEqual/y�
dropout_70/dropout/GreaterEqualGreaterEqual8dropout_70/dropout/random_uniform/RandomUniform:output:0*dropout_70/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2!
dropout_70/dropout/GreaterEqual�
dropout_70/dropout/CastCast#dropout_70/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2
dropout_70/dropout/Cast�
dropout_70/dropout/Mul_1Muldropout_70/dropout/Mul:z:0dropout_70/dropout/Cast:y:0*
T0*'
_output_shapes
:���������2
dropout_70/dropout/Mul_1�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulw
IdentityIdentitydropout_70/dropout/Mul_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^h_1/BiasAdd/ReadVariableOp^h_1/MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 28
h_1/BiasAdd/ReadVariableOph_1/BiasAdd/ReadVariableOp26
h_1/MatMul/ReadVariableOph_1/MatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�6
�
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543360

inputs#
seqblock_27543321:
seqblock_27543323:'
output_layer_27543326:#
output_layer_27543328:!
alphas_27543331:
alphas_27543333:%
distparam1_27543336:!
distparam1_27543338:%
distparam2_27543341:!
distparam2_27543343:
identity��>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp� SeqBlock/StatefulPartitionedCall�alphas/StatefulPartitionedCall�"distparam1/StatefulPartitionedCall�"distparam2/StatefulPartitionedCall�,h_1/kernel/Regularizer/Square/ReadVariableOp�$output_layer/StatefulPartitionedCall�
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinputsseqblock_27543321seqblock_27543323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_275432842"
 SeqBlock/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_27543326output_layer_27543328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_275430962&
$output_layer/StatefulPartitionedCall�
alphas/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0alphas_27543331alphas_27543333*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_alphas_layer_call_and_return_conditional_losses_275431132 
alphas/StatefulPartitionedCall�
"distparam1/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam1_27543336distparam1_27543338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_distparam1_layer_call_and_return_conditional_losses_275431302$
"distparam1/StatefulPartitionedCall�
"distparam2/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam2_27543341distparam2_27543343*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_distparam2_layer_call_and_return_conditional_losses_275431472$
"distparam2/StatefulPartitionedCall�
pvec/PartitionedCallPartitionedCall'alphas/StatefulPartitionedCall:output:0+distparam1/StatefulPartitionedCall:output:0+distparam2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_pvec_layer_call_and_return_conditional_losses_275431612
pvec/PartitionedCall�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_27543321*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_27543326*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/Square�
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Const�
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/Sum�
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:20
.MDN_size/output_layer/kernel/Regularizer/mul/x�
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp!^SeqBlock/StatefulPartitionedCall^alphas/StatefulPartitionedCall#^distparam1/StatefulPartitionedCall#^distparam2/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2@
alphas/StatefulPartitionedCallalphas/StatefulPartitionedCall2H
"distparam1/StatefulPartitionedCall"distparam1/StatefulPartitionedCall2H
"distparam2/StatefulPartitionedCall"distparam2/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_sequential_46_layer_call_fn_27543911

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_46_layer_call_and_return_conditional_losses_275430002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_distparam2_layer_call_and_return_conditional_losses_27543861

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_sequential_46_layer_call_fn_27542940
	h_1_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	h_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_46_layer_call_and_return_conditional_losses_275429332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������
#
_user_specified_name	h_1_input
�
�
&__inference_h_1_layer_call_fn_27543969

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_h_1_layer_call_and_return_conditional_losses_275429132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_output_layer_layer_call_fn_27543784

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_275430962
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
&__inference_signature_wrapper_27543537
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_275428892
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
A__inference_h_1_layer_call_and_return_conditional_losses_27542913

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,h_1/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
g
H__inference_dropout_70_layer_call_and_return_conditional_losses_27542960

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_distparam1_layer_call_and_return_conditional_losses_27543130

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_output_layer_layer_call_and_return_conditional_losses_27543096

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/Square�
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Const�
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/Sum�
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:20
.MDN_size/output_layer/kernel/Regularizer/mul/x�
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�6
�
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543450
input_1#
seqblock_27543411:
seqblock_27543413:'
output_layer_27543416:#
output_layer_27543418:!
alphas_27543421:
alphas_27543423:%
distparam1_27543426:!
distparam1_27543428:%
distparam2_27543431:!
distparam2_27543433:
identity��>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp� SeqBlock/StatefulPartitionedCall�alphas/StatefulPartitionedCall�"distparam1/StatefulPartitionedCall�"distparam2/StatefulPartitionedCall�,h_1/kernel/Regularizer/Square/ReadVariableOp�$output_layer/StatefulPartitionedCall�
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinput_1seqblock_27543411seqblock_27543413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_275430732"
 SeqBlock/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_27543416output_layer_27543418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_275430962&
$output_layer/StatefulPartitionedCall�
alphas/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0alphas_27543421alphas_27543423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_alphas_layer_call_and_return_conditional_losses_275431132 
alphas/StatefulPartitionedCall�
"distparam1/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam1_27543426distparam1_27543428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_distparam1_layer_call_and_return_conditional_losses_275431302$
"distparam1/StatefulPartitionedCall�
"distparam2/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam2_27543431distparam2_27543433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_distparam2_layer_call_and_return_conditional_losses_275431472$
"distparam2/StatefulPartitionedCall�
pvec/PartitionedCallPartitionedCall'alphas/StatefulPartitionedCall:output:0+distparam1/StatefulPartitionedCall:output:0+distparam2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_pvec_layer_call_and_return_conditional_losses_275431612
pvec/PartitionedCall�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_27543411*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_27543416*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/Square�
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Const�
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/Sum�
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:20
.MDN_size/output_layer/kernel/Regularizer/mul/x�
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp!^SeqBlock/StatefulPartitionedCall^alphas/StatefulPartitionedCall#^distparam1/StatefulPartitionedCall#^distparam2/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2@
alphas/StatefulPartitionedCallalphas/StatefulPartitionedCall2H
"distparam1/StatefulPartitionedCall"distparam1/StatefulPartitionedCall2H
"distparam2/StatefulPartitionedCall"distparam2/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543929

inputs4
"h_1_matmul_readvariableop_resource:1
#h_1_biasadd_readvariableop_resource:
identity��h_1/BiasAdd/ReadVariableOp�h_1/MatMul/ReadVariableOp�,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/MatMul/ReadVariableOpReadVariableOp"h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h_1/MatMul/ReadVariableOp

h_1/MatMulMatMulinputs!h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

h_1/MatMul�
h_1/BiasAdd/ReadVariableOpReadVariableOp#h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h_1/BiasAdd/ReadVariableOp�
h_1/BiasAddBiasAddh_1/MatMul:product:0"h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
h_1/BiasAddd
h_1/ReluReluh_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

h_1/Relu�
dropout_70/IdentityIdentityh_1/Relu:activations:0*
T0*'
_output_shapes
:���������2
dropout_70/Identity�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulw
IdentityIdentitydropout_70/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^h_1/BiasAdd/ReadVariableOp^h_1/MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 28
h_1/BiasAdd/ReadVariableOph_1/BiasAdd/ReadVariableOp26
h_1/MatMul/ReadVariableOph_1/MatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
-__inference_dropout_70_layer_call_fn_27543996

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_275429602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_27544024G
5h_1_kernel_regularizer_square_readvariableop_resource:
identity��,h_1/kernel/Regularizer/Square/ReadVariableOp�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5h_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
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
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp
�
f
H__inference_dropout_70_layer_call_and_return_conditional_losses_27542924

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_distparam2_layer_call_fn_27543850

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_distparam2_layer_call_and_return_conditional_losses_275431472
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�S
�
!__inference__traced_save_27544164
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
#savev2_h_1_bias_read_readvariableop$
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
*savev2_adam_h_1_bias_m_read_readvariableopB
>savev2_adam_mdn_size_output_layer_kernel_v_read_readvariableop@
<savev2_adam_mdn_size_output_layer_bias_v_read_readvariableop<
8savev2_adam_mdn_size_alphas_kernel_v_read_readvariableop:
6savev2_adam_mdn_size_alphas_bias_v_read_readvariableop@
<savev2_adam_mdn_size_distparam1_kernel_v_read_readvariableop>
:savev2_adam_mdn_size_distparam1_bias_v_read_readvariableop@
<savev2_adam_mdn_size_distparam2_kernel_v_read_readvariableop>
:savev2_adam_mdn_size_distparam2_bias_v_read_readvariableop0
,savev2_adam_h_1_kernel_v_read_readvariableop.
*savev2_adam_h_1_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB(alphas/kernel/.ATTRIBUTES/VARIABLE_VALUEB&alphas/bias/.ATTRIBUTES/VARIABLE_VALUEB,distparam1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*distparam1/bias/.ATTRIBUTES/VARIABLE_VALUEB,distparam2/kernel/.ATTRIBUTES/VARIABLE_VALUEB*distparam2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDalphas/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBalphas/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHdistparam1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFdistparam1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHdistparam2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFdistparam2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDalphas/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBalphas/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHdistparam1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFdistparam1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHdistparam2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFdistparam2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_mdn_size_output_layer_kernel_read_readvariableop5savev2_mdn_size_output_layer_bias_read_readvariableop1savev2_mdn_size_alphas_kernel_read_readvariableop/savev2_mdn_size_alphas_bias_read_readvariableop5savev2_mdn_size_distparam1_kernel_read_readvariableop3savev2_mdn_size_distparam1_bias_read_readvariableop5savev2_mdn_size_distparam2_kernel_read_readvariableop3savev2_mdn_size_distparam2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop%savev2_h_1_kernel_read_readvariableop#savev2_h_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop>savev2_adam_mdn_size_output_layer_kernel_m_read_readvariableop<savev2_adam_mdn_size_output_layer_bias_m_read_readvariableop8savev2_adam_mdn_size_alphas_kernel_m_read_readvariableop6savev2_adam_mdn_size_alphas_bias_m_read_readvariableop<savev2_adam_mdn_size_distparam1_kernel_m_read_readvariableop:savev2_adam_mdn_size_distparam1_bias_m_read_readvariableop<savev2_adam_mdn_size_distparam2_kernel_m_read_readvariableop:savev2_adam_mdn_size_distparam2_bias_m_read_readvariableop,savev2_adam_h_1_kernel_m_read_readvariableop*savev2_adam_h_1_bias_m_read_readvariableop>savev2_adam_mdn_size_output_layer_kernel_v_read_readvariableop<savev2_adam_mdn_size_output_layer_bias_v_read_readvariableop8savev2_adam_mdn_size_alphas_kernel_v_read_readvariableop6savev2_adam_mdn_size_alphas_bias_v_read_readvariableop<savev2_adam_mdn_size_distparam1_kernel_v_read_readvariableop:savev2_adam_mdn_size_distparam1_bias_v_read_readvariableop<savev2_adam_mdn_size_distparam2_kernel_v_read_readvariableop:savev2_adam_mdn_size_distparam2_bias_v_read_readvariableop,savev2_adam_h_1_kernel_v_read_readvariableop*savev2_adam_h_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::::: : : : : ::: : : : ::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 
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

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

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

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::(

_output_shapes
: 
�^
�	
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543702

inputsK
9seqblock_sequential_46_h_1_matmul_readvariableop_resource:H
:seqblock_sequential_46_h_1_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:7
%alphas_matmul_readvariableop_resource:4
&alphas_biasadd_readvariableop_resource:;
)distparam1_matmul_readvariableop_resource:8
*distparam1_biasadd_readvariableop_resource:;
)distparam2_matmul_readvariableop_resource:8
*distparam2_biasadd_readvariableop_resource:
identity��>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�1SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp�0SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp�alphas/BiasAdd/ReadVariableOp�alphas/MatMul/ReadVariableOp�!distparam1/BiasAdd/ReadVariableOp� distparam1/MatMul/ReadVariableOp�!distparam2/BiasAdd/ReadVariableOp� distparam2/MatMul/ReadVariableOp�,h_1/kernel/Regularizer/Square/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
0SeqBlock/sequential_46/h_1/MatMul/ReadVariableOpReadVariableOp9seqblock_sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype022
0SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp�
!SeqBlock/sequential_46/h_1/MatMulMatMulinputs8SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!SeqBlock/sequential_46/h_1/MatMul�
1SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOpReadVariableOp:seqblock_sequential_46_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp�
"SeqBlock/sequential_46/h_1/BiasAddBiasAdd+SeqBlock/sequential_46/h_1/MatMul:product:09SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"SeqBlock/sequential_46/h_1/BiasAdd�
SeqBlock/sequential_46/h_1/ReluRelu+SeqBlock/sequential_46/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2!
SeqBlock/sequential_46/h_1/Relu�
/SeqBlock/sequential_46/dropout_70/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?21
/SeqBlock/sequential_46/dropout_70/dropout/Const�
-SeqBlock/sequential_46/dropout_70/dropout/MulMul-SeqBlock/sequential_46/h_1/Relu:activations:08SeqBlock/sequential_46/dropout_70/dropout/Const:output:0*
T0*'
_output_shapes
:���������2/
-SeqBlock/sequential_46/dropout_70/dropout/Mul�
/SeqBlock/sequential_46/dropout_70/dropout/ShapeShape-SeqBlock/sequential_46/h_1/Relu:activations:0*
T0*
_output_shapes
:21
/SeqBlock/sequential_46/dropout_70/dropout/Shape�
FSeqBlock/sequential_46/dropout_70/dropout/random_uniform/RandomUniformRandomUniform8SeqBlock/sequential_46/dropout_70/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed�2H
FSeqBlock/sequential_46/dropout_70/dropout/random_uniform/RandomUniform�
8SeqBlock/sequential_46/dropout_70/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2:
8SeqBlock/sequential_46/dropout_70/dropout/GreaterEqual/y�
6SeqBlock/sequential_46/dropout_70/dropout/GreaterEqualGreaterEqualOSeqBlock/sequential_46/dropout_70/dropout/random_uniform/RandomUniform:output:0ASeqBlock/sequential_46/dropout_70/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������28
6SeqBlock/sequential_46/dropout_70/dropout/GreaterEqual�
.SeqBlock/sequential_46/dropout_70/dropout/CastCast:SeqBlock/sequential_46/dropout_70/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������20
.SeqBlock/sequential_46/dropout_70/dropout/Cast�
/SeqBlock/sequential_46/dropout_70/dropout/Mul_1Mul1SeqBlock/sequential_46/dropout_70/dropout/Mul:z:02SeqBlock/sequential_46/dropout_70/dropout/Cast:y:0*
T0*'
_output_shapes
:���������21
/SeqBlock/sequential_46/dropout_70/dropout/Mul_1�
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"output_layer/MatMul/ReadVariableOp�
output_layer/MatMulMatMul3SeqBlock/sequential_46/dropout_70/dropout/Mul_1:z:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output_layer/MatMul�
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOp�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output_layer/BiasAdd
output_layer/ReluReluoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
output_layer/Relu�
alphas/MatMul/ReadVariableOpReadVariableOp%alphas_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
alphas/MatMul/ReadVariableOp�
alphas/MatMulMatMuloutput_layer/Relu:activations:0$alphas/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
alphas/MatMul�
alphas/BiasAdd/ReadVariableOpReadVariableOp&alphas_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
alphas/BiasAdd/ReadVariableOp�
alphas/BiasAddBiasAddalphas/MatMul:product:0%alphas/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
alphas/BiasAddv
alphas/SoftmaxSoftmaxalphas/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
alphas/Softmax�
 distparam1/MatMul/ReadVariableOpReadVariableOp)distparam1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 distparam1/MatMul/ReadVariableOp�
distparam1/MatMulMatMuloutput_layer/Relu:activations:0(distparam1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
distparam1/MatMul�
!distparam1/BiasAdd/ReadVariableOpReadVariableOp*distparam1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!distparam1/BiasAdd/ReadVariableOp�
distparam1/BiasAddBiasAdddistparam1/MatMul:product:0)distparam1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
distparam1/BiasAdd�
distparam1/SoftplusSoftplusdistparam1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
distparam1/Softplus�
 distparam2/MatMul/ReadVariableOpReadVariableOp)distparam2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 distparam2/MatMul/ReadVariableOp�
distparam2/MatMulMatMuloutput_layer/Relu:activations:0(distparam2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
distparam2/MatMul�
!distparam2/BiasAdd/ReadVariableOpReadVariableOp*distparam2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!distparam2/BiasAdd/ReadVariableOp�
distparam2/BiasAddBiasAdddistparam2/MatMul:product:0)distparam2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
distparam2/BiasAdd�
distparam2/SoftplusSoftplusdistparam2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
distparam2/Softplusf
pvec/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
pvec/concat/axis�
pvec/concatConcatV2alphas/Softmax:softmax:0!distparam1/Softplus:activations:0!distparam2/Softplus:activations:0pvec/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
pvec/concat�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9seqblock_sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/Square�
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Const�
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/Sum�
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:20
.MDN_size/output_layer/kernel/Regularizer/mul/x�
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulo
IdentityIdentitypvec/concat:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2^SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp1^SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp^alphas/BiasAdd/ReadVariableOp^alphas/MatMul/ReadVariableOp"^distparam1/BiasAdd/ReadVariableOp!^distparam1/MatMul/ReadVariableOp"^distparam2/BiasAdd/ReadVariableOp!^distparam2/MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2f
1SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp1SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp2d
0SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp0SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp2>
alphas/BiasAdd/ReadVariableOpalphas/BiasAdd/ReadVariableOp2<
alphas/MatMul/ReadVariableOpalphas/MatMul/ReadVariableOp2F
!distparam1/BiasAdd/ReadVariableOp!distparam1/BiasAdd/ReadVariableOp2D
 distparam1/MatMul/ReadVariableOp distparam1/MatMul/ReadVariableOp2F
!distparam2/BiasAdd/ReadVariableOp!distparam2/BiasAdd/ReadVariableOp2D
 distparam2/MatMul/ReadVariableOp distparam2/MatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
g
H__inference_dropout_70_layer_call_and_return_conditional_losses_27544013

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed�2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_distparam1_layer_call_and_return_conditional_losses_27543841

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_sequential_46_layer_call_fn_27543016
	h_1_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	h_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_46_layer_call_and_return_conditional_losses_275430002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������
#
_user_specified_name	h_1_input
�

�
+__inference_MDN_size_layer_call_fn_27543562

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_MDN_size_layer_call_and_return_conditional_losses_275431762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�6
�
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543492
input_1#
seqblock_27543453:
seqblock_27543455:'
output_layer_27543458:#
output_layer_27543460:!
alphas_27543463:
alphas_27543465:%
distparam1_27543468:!
distparam1_27543470:%
distparam2_27543473:!
distparam2_27543475:
identity��>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp� SeqBlock/StatefulPartitionedCall�alphas/StatefulPartitionedCall�"distparam1/StatefulPartitionedCall�"distparam2/StatefulPartitionedCall�,h_1/kernel/Regularizer/Square/ReadVariableOp�$output_layer/StatefulPartitionedCall�
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinput_1seqblock_27543453seqblock_27543455*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_275432842"
 SeqBlock/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_27543458output_layer_27543460*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_275430962&
$output_layer/StatefulPartitionedCall�
alphas/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0alphas_27543463alphas_27543465*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_alphas_layer_call_and_return_conditional_losses_275431132 
alphas/StatefulPartitionedCall�
"distparam1/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam1_27543468distparam1_27543470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_distparam1_layer_call_and_return_conditional_losses_275431302$
"distparam1/StatefulPartitionedCall�
"distparam2/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam2_27543473distparam2_27543475*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_distparam2_layer_call_and_return_conditional_losses_275431472$
"distparam2/StatefulPartitionedCall�
pvec/PartitionedCallPartitionedCall'alphas/StatefulPartitionedCall:output:0+distparam1/StatefulPartitionedCall:output:0+distparam2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_pvec_layer_call_and_return_conditional_losses_275431612
pvec/PartitionedCall�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_27543453*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_27543458*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/Square�
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Const�
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/Sum�
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:20
.MDN_size/output_layer/kernel/Regularizer/mul/x�
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp!^SeqBlock/StatefulPartitionedCall^alphas/StatefulPartitionedCall#^distparam1/StatefulPartitionedCall#^distparam2/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2@
alphas/StatefulPartitionedCallalphas/StatefulPartitionedCall2H
"distparam1/StatefulPartitionedCall"distparam1/StatefulPartitionedCall2H
"distparam2/StatefulPartitionedCall"distparam2/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
|
B__inference_pvec_layer_call_and_return_conditional_losses_27543876
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�
I
-__inference_dropout_70_layer_call_fn_27543991

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_275429242
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_SeqBlock_layer_call_and_return_conditional_losses_27543073

inputsB
0sequential_46_h_1_matmul_readvariableop_resource:?
1sequential_46_h_1_biasadd_readvariableop_resource:
identity��,h_1/kernel/Regularizer/Square/ReadVariableOp�(sequential_46/h_1/BiasAdd/ReadVariableOp�'sequential_46/h_1/MatMul/ReadVariableOp�
'sequential_46/h_1/MatMul/ReadVariableOpReadVariableOp0sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_46/h_1/MatMul/ReadVariableOp�
sequential_46/h_1/MatMulMatMulinputs/sequential_46/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/MatMul�
(sequential_46/h_1/BiasAdd/ReadVariableOpReadVariableOp1sequential_46_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_46/h_1/BiasAdd/ReadVariableOp�
sequential_46/h_1/BiasAddBiasAdd"sequential_46/h_1/MatMul:product:00sequential_46/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/BiasAdd�
sequential_46/h_1/ReluRelu"sequential_46/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/Relu�
!sequential_46/dropout_70/IdentityIdentity$sequential_46/h_1/Relu:activations:0*
T0*'
_output_shapes
:���������2#
!sequential_46/dropout_70/Identity�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
IdentityIdentity*sequential_46/dropout_70/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp-^h_1/kernel/Regularizer/Square/ReadVariableOp)^sequential_46/h_1/BiasAdd/ReadVariableOp(^sequential_46/h_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2T
(sequential_46/h_1/BiasAdd/ReadVariableOp(sequential_46/h_1/BiasAdd/ReadVariableOp2R
'sequential_46/h_1/MatMul/ReadVariableOp'sequential_46/h_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543000

inputs
h_1_27542987:
h_1_27542989:
identity��"dropout_70/StatefulPartitionedCall�h_1/StatefulPartitionedCall�,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/StatefulPartitionedCallStatefulPartitionedCallinputsh_1_27542987h_1_27542989*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_h_1_layer_call_and_return_conditional_losses_275429132
h_1/StatefulPartitionedCall�
"dropout_70/StatefulPartitionedCallStatefulPartitionedCall$h_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_275429602$
"dropout_70/StatefulPartitionedCall�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_1_27542987*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
IdentityIdentity+dropout_70/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp#^dropout_70/StatefulPartitionedCall^h_1/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2H
"dropout_70/StatefulPartitionedCall"dropout_70/StatefulPartitionedCall2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
'__inference_pvec_layer_call_fn_27543868
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_pvec_layer_call_and_return_conditional_losses_275431612
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�
z
B__inference_pvec_layer_call_and_return_conditional_losses_27543161

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543048
	h_1_input
h_1_27543035:
h_1_27543037:
identity��"dropout_70/StatefulPartitionedCall�h_1/StatefulPartitionedCall�,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/StatefulPartitionedCallStatefulPartitionedCall	h_1_inputh_1_27543035h_1_27543037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_h_1_layer_call_and_return_conditional_losses_275429132
h_1/StatefulPartitionedCall�
"dropout_70/StatefulPartitionedCallStatefulPartitionedCall$h_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_275429602$
"dropout_70/StatefulPartitionedCall�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_1_27543035*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
IdentityIdentity+dropout_70/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp#^dropout_70/StatefulPartitionedCall^h_1/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2H
"dropout_70/StatefulPartitionedCall"dropout_70/StatefulPartitionedCall2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp:R N
'
_output_shapes
:���������
#
_user_specified_name	h_1_input
�
�
D__inference_alphas_layer_call_and_return_conditional_losses_27543821

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
+__inference_MDN_size_layer_call_fn_27543199
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_MDN_size_layer_call_and_return_conditional_losses_275431762
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
D__inference_alphas_layer_call_and_return_conditional_losses_27543113

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_SeqBlock_layer_call_and_return_conditional_losses_27543744

inputsB
0sequential_46_h_1_matmul_readvariableop_resource:?
1sequential_46_h_1_biasadd_readvariableop_resource:
identity��,h_1/kernel/Regularizer/Square/ReadVariableOp�(sequential_46/h_1/BiasAdd/ReadVariableOp�'sequential_46/h_1/MatMul/ReadVariableOp�
'sequential_46/h_1/MatMul/ReadVariableOpReadVariableOp0sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_46/h_1/MatMul/ReadVariableOp�
sequential_46/h_1/MatMulMatMulinputs/sequential_46/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/MatMul�
(sequential_46/h_1/BiasAdd/ReadVariableOpReadVariableOp1sequential_46_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_46/h_1/BiasAdd/ReadVariableOp�
sequential_46/h_1/BiasAddBiasAdd"sequential_46/h_1/MatMul:product:00sequential_46/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/BiasAdd�
sequential_46/h_1/ReluRelu"sequential_46/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/Relu�
!sequential_46/dropout_70/IdentityIdentity$sequential_46/h_1/Relu:activations:0*
T0*'
_output_shapes
:���������2#
!sequential_46/dropout_70/Identity�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
IdentityIdentity*sequential_46/dropout_70/Identity:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp-^h_1/kernel/Regularizer/Square/ReadVariableOp)^sequential_46/h_1/BiasAdd/ReadVariableOp(^sequential_46/h_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2T
(sequential_46/h_1/BiasAdd/ReadVariableOp(sequential_46/h_1/BiasAdd/ReadVariableOp2R
'sequential_46/h_1/MatMul/ReadVariableOp'sequential_46/h_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_27543887Y
Gmdn_size_output_layer_kernel_regularizer_square_readvariableop_resource:
identity��>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpGmdn_size_output_layer_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/Square�
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Const�
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/Sum�
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:20
.MDN_size/output_layer/kernel/Regularizer/mul/x�
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulz
IdentityIdentity0MDN_size/output_layer/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identity�
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp
�%
�
F__inference_SeqBlock_layer_call_and_return_conditional_losses_27543769

inputsB
0sequential_46_h_1_matmul_readvariableop_resource:?
1sequential_46_h_1_biasadd_readvariableop_resource:
identity��,h_1/kernel/Regularizer/Square/ReadVariableOp�(sequential_46/h_1/BiasAdd/ReadVariableOp�'sequential_46/h_1/MatMul/ReadVariableOp�
'sequential_46/h_1/MatMul/ReadVariableOpReadVariableOp0sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_46/h_1/MatMul/ReadVariableOp�
sequential_46/h_1/MatMulMatMulinputs/sequential_46/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/MatMul�
(sequential_46/h_1/BiasAdd/ReadVariableOpReadVariableOp1sequential_46_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_46/h_1/BiasAdd/ReadVariableOp�
sequential_46/h_1/BiasAddBiasAdd"sequential_46/h_1/MatMul:product:00sequential_46/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/BiasAdd�
sequential_46/h_1/ReluRelu"sequential_46/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/Relu�
&sequential_46/dropout_70/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2(
&sequential_46/dropout_70/dropout/Const�
$sequential_46/dropout_70/dropout/MulMul$sequential_46/h_1/Relu:activations:0/sequential_46/dropout_70/dropout/Const:output:0*
T0*'
_output_shapes
:���������2&
$sequential_46/dropout_70/dropout/Mul�
&sequential_46/dropout_70/dropout/ShapeShape$sequential_46/h_1/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_46/dropout_70/dropout/Shape�
=sequential_46/dropout_70/dropout/random_uniform/RandomUniformRandomUniform/sequential_46/dropout_70/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed�2?
=sequential_46/dropout_70/dropout/random_uniform/RandomUniform�
/sequential_46/dropout_70/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>21
/sequential_46/dropout_70/dropout/GreaterEqual/y�
-sequential_46/dropout_70/dropout/GreaterEqualGreaterEqualFsequential_46/dropout_70/dropout/random_uniform/RandomUniform:output:08sequential_46/dropout_70/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2/
-sequential_46/dropout_70/dropout/GreaterEqual�
%sequential_46/dropout_70/dropout/CastCast1sequential_46/dropout_70/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2'
%sequential_46/dropout_70/dropout/Cast�
&sequential_46/dropout_70/dropout/Mul_1Mul(sequential_46/dropout_70/dropout/Mul:z:0)sequential_46/dropout_70/dropout/Cast:y:0*
T0*'
_output_shapes
:���������2(
&sequential_46/dropout_70/dropout/Mul_1�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
IdentityIdentity*sequential_46/dropout_70/dropout/Mul_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp-^h_1/kernel/Regularizer/Square/ReadVariableOp)^sequential_46/h_1/BiasAdd/ReadVariableOp(^sequential_46/h_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2T
(sequential_46/h_1/BiasAdd/ReadVariableOp(sequential_46/h_1/BiasAdd/ReadVariableOp2R
'sequential_46/h_1/MatMul/ReadVariableOp'sequential_46/h_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
+__inference_MDN_size_layer_call_fn_27543408
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_MDN_size_layer_call_and_return_conditional_losses_275433602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�P
�	
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543641

inputsK
9seqblock_sequential_46_h_1_matmul_readvariableop_resource:H
:seqblock_sequential_46_h_1_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:7
%alphas_matmul_readvariableop_resource:4
&alphas_biasadd_readvariableop_resource:;
)distparam1_matmul_readvariableop_resource:8
*distparam1_biasadd_readvariableop_resource:;
)distparam2_matmul_readvariableop_resource:8
*distparam2_biasadd_readvariableop_resource:
identity��>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�1SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp�0SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp�alphas/BiasAdd/ReadVariableOp�alphas/MatMul/ReadVariableOp�!distparam1/BiasAdd/ReadVariableOp� distparam1/MatMul/ReadVariableOp�!distparam2/BiasAdd/ReadVariableOp� distparam2/MatMul/ReadVariableOp�,h_1/kernel/Regularizer/Square/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�
0SeqBlock/sequential_46/h_1/MatMul/ReadVariableOpReadVariableOp9seqblock_sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype022
0SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp�
!SeqBlock/sequential_46/h_1/MatMulMatMulinputs8SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2#
!SeqBlock/sequential_46/h_1/MatMul�
1SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOpReadVariableOp:seqblock_sequential_46_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp�
"SeqBlock/sequential_46/h_1/BiasAddBiasAdd+SeqBlock/sequential_46/h_1/MatMul:product:09SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2$
"SeqBlock/sequential_46/h_1/BiasAdd�
SeqBlock/sequential_46/h_1/ReluRelu+SeqBlock/sequential_46/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2!
SeqBlock/sequential_46/h_1/Relu�
*SeqBlock/sequential_46/dropout_70/IdentityIdentity-SeqBlock/sequential_46/h_1/Relu:activations:0*
T0*'
_output_shapes
:���������2,
*SeqBlock/sequential_46/dropout_70/Identity�
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"output_layer/MatMul/ReadVariableOp�
output_layer/MatMulMatMul3SeqBlock/sequential_46/dropout_70/Identity:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output_layer/MatMul�
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOp�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output_layer/BiasAdd
output_layer/ReluReluoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
output_layer/Relu�
alphas/MatMul/ReadVariableOpReadVariableOp%alphas_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
alphas/MatMul/ReadVariableOp�
alphas/MatMulMatMuloutput_layer/Relu:activations:0$alphas/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
alphas/MatMul�
alphas/BiasAdd/ReadVariableOpReadVariableOp&alphas_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
alphas/BiasAdd/ReadVariableOp�
alphas/BiasAddBiasAddalphas/MatMul:product:0%alphas/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
alphas/BiasAddv
alphas/SoftmaxSoftmaxalphas/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
alphas/Softmax�
 distparam1/MatMul/ReadVariableOpReadVariableOp)distparam1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 distparam1/MatMul/ReadVariableOp�
distparam1/MatMulMatMuloutput_layer/Relu:activations:0(distparam1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
distparam1/MatMul�
!distparam1/BiasAdd/ReadVariableOpReadVariableOp*distparam1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!distparam1/BiasAdd/ReadVariableOp�
distparam1/BiasAddBiasAdddistparam1/MatMul:product:0)distparam1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
distparam1/BiasAdd�
distparam1/SoftplusSoftplusdistparam1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
distparam1/Softplus�
 distparam2/MatMul/ReadVariableOpReadVariableOp)distparam2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 distparam2/MatMul/ReadVariableOp�
distparam2/MatMulMatMuloutput_layer/Relu:activations:0(distparam2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
distparam2/MatMul�
!distparam2/BiasAdd/ReadVariableOpReadVariableOp*distparam2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!distparam2/BiasAdd/ReadVariableOp�
distparam2/BiasAddBiasAdddistparam2/MatMul:product:0)distparam2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
distparam2/BiasAdd�
distparam2/SoftplusSoftplusdistparam2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
distparam2/Softplusf
pvec/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
pvec/concat/axis�
pvec/concatConcatV2alphas/Softmax:softmax:0!distparam1/Softplus:activations:0!distparam2/Softplus:activations:0pvec/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
pvec/concat�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9seqblock_sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/Square�
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Const�
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/Sum�
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:20
.MDN_size/output_layer/kernel/Regularizer/mul/x�
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulo
IdentityIdentitypvec/concat:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2^SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp1^SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp^alphas/BiasAdd/ReadVariableOp^alphas/MatMul/ReadVariableOp"^distparam1/BiasAdd/ReadVariableOp!^distparam1/MatMul/ReadVariableOp"^distparam2/BiasAdd/ReadVariableOp!^distparam2/MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2f
1SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp1SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp2d
0SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp0SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp2>
alphas/BiasAdd/ReadVariableOpalphas/BiasAdd/ReadVariableOp2<
alphas/MatMul/ReadVariableOpalphas/MatMul/ReadVariableOp2F
!distparam1/BiasAdd/ReadVariableOp!distparam1/BiasAdd/ReadVariableOp2D
 distparam1/MatMul/ReadVariableOp distparam1/MatMul/ReadVariableOp2F
!distparam2/BiasAdd/ReadVariableOp!distparam2/BiasAdd/ReadVariableOp2D
 distparam2/MatMul/ReadVariableOp distparam2/MatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
+__inference_MDN_size_layer_call_fn_27543587

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_MDN_size_layer_call_and_return_conditional_losses_275433602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_h_1_layer_call_and_return_conditional_losses_27543986

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�,h_1/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^h_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_SeqBlock_layer_call_fn_27543726

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_275432842
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�E
�	
#__inference__wrapped_model_27542889
input_1T
Bmdn_size_seqblock_sequential_46_h_1_matmul_readvariableop_resource:Q
Cmdn_size_seqblock_sequential_46_h_1_biasadd_readvariableop_resource:F
4mdn_size_output_layer_matmul_readvariableop_resource:C
5mdn_size_output_layer_biasadd_readvariableop_resource:@
.mdn_size_alphas_matmul_readvariableop_resource:=
/mdn_size_alphas_biasadd_readvariableop_resource:D
2mdn_size_distparam1_matmul_readvariableop_resource:A
3mdn_size_distparam1_biasadd_readvariableop_resource:D
2mdn_size_distparam2_matmul_readvariableop_resource:A
3mdn_size_distparam2_biasadd_readvariableop_resource:
identity��:MDN_size/SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp�9MDN_size/SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp�&MDN_size/alphas/BiasAdd/ReadVariableOp�%MDN_size/alphas/MatMul/ReadVariableOp�*MDN_size/distparam1/BiasAdd/ReadVariableOp�)MDN_size/distparam1/MatMul/ReadVariableOp�*MDN_size/distparam2/BiasAdd/ReadVariableOp�)MDN_size/distparam2/MatMul/ReadVariableOp�,MDN_size/output_layer/BiasAdd/ReadVariableOp�+MDN_size/output_layer/MatMul/ReadVariableOp�
9MDN_size/SeqBlock/sequential_46/h_1/MatMul/ReadVariableOpReadVariableOpBmdn_size_seqblock_sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02;
9MDN_size/SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp�
*MDN_size/SeqBlock/sequential_46/h_1/MatMulMatMulinput_1AMDN_size/SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2,
*MDN_size/SeqBlock/sequential_46/h_1/MatMul�
:MDN_size/SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOpReadVariableOpCmdn_size_seqblock_sequential_46_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:MDN_size/SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp�
+MDN_size/SeqBlock/sequential_46/h_1/BiasAddBiasAdd4MDN_size/SeqBlock/sequential_46/h_1/MatMul:product:0BMDN_size/SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2-
+MDN_size/SeqBlock/sequential_46/h_1/BiasAdd�
(MDN_size/SeqBlock/sequential_46/h_1/ReluRelu4MDN_size/SeqBlock/sequential_46/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2*
(MDN_size/SeqBlock/sequential_46/h_1/Relu�
3MDN_size/SeqBlock/sequential_46/dropout_70/IdentityIdentity6MDN_size/SeqBlock/sequential_46/h_1/Relu:activations:0*
T0*'
_output_shapes
:���������25
3MDN_size/SeqBlock/sequential_46/dropout_70/Identity�
+MDN_size/output_layer/MatMul/ReadVariableOpReadVariableOp4mdn_size_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+MDN_size/output_layer/MatMul/ReadVariableOp�
MDN_size/output_layer/MatMulMatMul<MDN_size/SeqBlock/sequential_46/dropout_70/Identity:output:03MDN_size/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MDN_size/output_layer/MatMul�
,MDN_size/output_layer/BiasAdd/ReadVariableOpReadVariableOp5mdn_size_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,MDN_size/output_layer/BiasAdd/ReadVariableOp�
MDN_size/output_layer/BiasAddBiasAdd&MDN_size/output_layer/MatMul:product:04MDN_size/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MDN_size/output_layer/BiasAdd�
MDN_size/output_layer/ReluRelu&MDN_size/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
MDN_size/output_layer/Relu�
%MDN_size/alphas/MatMul/ReadVariableOpReadVariableOp.mdn_size_alphas_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%MDN_size/alphas/MatMul/ReadVariableOp�
MDN_size/alphas/MatMulMatMul(MDN_size/output_layer/Relu:activations:0-MDN_size/alphas/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MDN_size/alphas/MatMul�
&MDN_size/alphas/BiasAdd/ReadVariableOpReadVariableOp/mdn_size_alphas_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&MDN_size/alphas/BiasAdd/ReadVariableOp�
MDN_size/alphas/BiasAddBiasAdd MDN_size/alphas/MatMul:product:0.MDN_size/alphas/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MDN_size/alphas/BiasAdd�
MDN_size/alphas/SoftmaxSoftmax MDN_size/alphas/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
MDN_size/alphas/Softmax�
)MDN_size/distparam1/MatMul/ReadVariableOpReadVariableOp2mdn_size_distparam1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)MDN_size/distparam1/MatMul/ReadVariableOp�
MDN_size/distparam1/MatMulMatMul(MDN_size/output_layer/Relu:activations:01MDN_size/distparam1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MDN_size/distparam1/MatMul�
*MDN_size/distparam1/BiasAdd/ReadVariableOpReadVariableOp3mdn_size_distparam1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*MDN_size/distparam1/BiasAdd/ReadVariableOp�
MDN_size/distparam1/BiasAddBiasAdd$MDN_size/distparam1/MatMul:product:02MDN_size/distparam1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MDN_size/distparam1/BiasAdd�
MDN_size/distparam1/SoftplusSoftplus$MDN_size/distparam1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
MDN_size/distparam1/Softplus�
)MDN_size/distparam2/MatMul/ReadVariableOpReadVariableOp2mdn_size_distparam2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)MDN_size/distparam2/MatMul/ReadVariableOp�
MDN_size/distparam2/MatMulMatMul(MDN_size/output_layer/Relu:activations:01MDN_size/distparam2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MDN_size/distparam2/MatMul�
*MDN_size/distparam2/BiasAdd/ReadVariableOpReadVariableOp3mdn_size_distparam2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*MDN_size/distparam2/BiasAdd/ReadVariableOp�
MDN_size/distparam2/BiasAddBiasAdd$MDN_size/distparam2/MatMul:product:02MDN_size/distparam2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MDN_size/distparam2/BiasAdd�
MDN_size/distparam2/SoftplusSoftplus$MDN_size/distparam2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
MDN_size/distparam2/Softplusx
MDN_size/pvec/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
MDN_size/pvec/concat/axis�
MDN_size/pvec/concatConcatV2!MDN_size/alphas/Softmax:softmax:0*MDN_size/distparam1/Softplus:activations:0*MDN_size/distparam2/Softplus:activations:0"MDN_size/pvec/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
MDN_size/pvec/concatx
IdentityIdentityMDN_size/pvec/concat:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp;^MDN_size/SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp:^MDN_size/SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp'^MDN_size/alphas/BiasAdd/ReadVariableOp&^MDN_size/alphas/MatMul/ReadVariableOp+^MDN_size/distparam1/BiasAdd/ReadVariableOp*^MDN_size/distparam1/MatMul/ReadVariableOp+^MDN_size/distparam2/BiasAdd/ReadVariableOp*^MDN_size/distparam2/MatMul/ReadVariableOp-^MDN_size/output_layer/BiasAdd/ReadVariableOp,^MDN_size/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2x
:MDN_size/SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp:MDN_size/SeqBlock/sequential_46/h_1/BiasAdd/ReadVariableOp2v
9MDN_size/SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp9MDN_size/SeqBlock/sequential_46/h_1/MatMul/ReadVariableOp2P
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
:���������
!
_user_specified_name	input_1
�6
�
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543176

inputs#
seqblock_27543074:
seqblock_27543076:'
output_layer_27543097:#
output_layer_27543099:!
alphas_27543114:
alphas_27543116:%
distparam1_27543131:!
distparam1_27543133:%
distparam2_27543148:!
distparam2_27543150:
identity��>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp� SeqBlock/StatefulPartitionedCall�alphas/StatefulPartitionedCall�"distparam1/StatefulPartitionedCall�"distparam2/StatefulPartitionedCall�,h_1/kernel/Regularizer/Square/ReadVariableOp�$output_layer/StatefulPartitionedCall�
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinputsseqblock_27543074seqblock_27543076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_SeqBlock_layer_call_and_return_conditional_losses_275430732"
 SeqBlock/StatefulPartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_27543097output_layer_27543099*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_output_layer_layer_call_and_return_conditional_losses_275430962&
$output_layer/StatefulPartitionedCall�
alphas/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0alphas_27543114alphas_27543116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_alphas_layer_call_and_return_conditional_losses_275431132 
alphas/StatefulPartitionedCall�
"distparam1/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam1_27543131distparam1_27543133*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_distparam1_layer_call_and_return_conditional_losses_275431302$
"distparam1/StatefulPartitionedCall�
"distparam2/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0distparam2_27543148distparam2_27543150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_distparam2_layer_call_and_return_conditional_losses_275431472$
"distparam2/StatefulPartitionedCall�
pvec/PartitionedCallPartitionedCall'alphas/StatefulPartitionedCall:output:0+distparam1/StatefulPartitionedCall:output:0+distparam2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_pvec_layer_call_and_return_conditional_losses_275431612
pvec/PartitionedCall�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpseqblock_27543074*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpoutput_layer_27543097*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/Square�
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Const�
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/Sum�
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:20
.MDN_size/output_layer/kernel/Regularizer/mul/x�
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp!^SeqBlock/StatefulPartitionedCall^alphas/StatefulPartitionedCall#^distparam1/StatefulPartitionedCall#^distparam2/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : : : 2�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2@
alphas/StatefulPartitionedCallalphas/StatefulPartitionedCall2H
"distparam1/StatefulPartitionedCall"distparam1/StatefulPartitionedCall2H
"distparam2/StatefulPartitionedCall"distparam2/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_distparam2_layer_call_and_return_conditional_losses_27543147

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_distparam1_layer_call_fn_27543830

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_distparam1_layer_call_and_return_conditional_losses_275431302
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
H__inference_dropout_70_layer_call_and_return_conditional_losses_27544001

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543032
	h_1_input
h_1_27543019:
h_1_27543021:
identity��h_1/StatefulPartitionedCall�,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/StatefulPartitionedCallStatefulPartitionedCall	h_1_inputh_1_27543019h_1_27543021*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_h_1_layer_call_and_return_conditional_losses_275429132
h_1/StatefulPartitionedCall�
dropout_70/PartitionedCallPartitionedCall$h_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_275429242
dropout_70/PartitionedCall�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_1_27543019*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul~
IdentityIdentity#dropout_70/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^h_1/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp:R N
'
_output_shapes
:���������
#
_user_specified_name	h_1_input
�%
�
F__inference_SeqBlock_layer_call_and_return_conditional_losses_27543284

inputsB
0sequential_46_h_1_matmul_readvariableop_resource:?
1sequential_46_h_1_biasadd_readvariableop_resource:
identity��,h_1/kernel/Regularizer/Square/ReadVariableOp�(sequential_46/h_1/BiasAdd/ReadVariableOp�'sequential_46/h_1/MatMul/ReadVariableOp�
'sequential_46/h_1/MatMul/ReadVariableOpReadVariableOp0sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_46/h_1/MatMul/ReadVariableOp�
sequential_46/h_1/MatMulMatMulinputs/sequential_46/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/MatMul�
(sequential_46/h_1/BiasAdd/ReadVariableOpReadVariableOp1sequential_46_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_46/h_1/BiasAdd/ReadVariableOp�
sequential_46/h_1/BiasAddBiasAdd"sequential_46/h_1/MatMul:product:00sequential_46/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/BiasAdd�
sequential_46/h_1/ReluRelu"sequential_46/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential_46/h_1/Relu�
&sequential_46/dropout_70/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2(
&sequential_46/dropout_70/dropout/Const�
$sequential_46/dropout_70/dropout/MulMul$sequential_46/h_1/Relu:activations:0/sequential_46/dropout_70/dropout/Const:output:0*
T0*'
_output_shapes
:���������2&
$sequential_46/dropout_70/dropout/Mul�
&sequential_46/dropout_70/dropout/ShapeShape$sequential_46/h_1/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_46/dropout_70/dropout/Shape�
=sequential_46/dropout_70/dropout/random_uniform/RandomUniformRandomUniform/sequential_46/dropout_70/dropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed�2?
=sequential_46/dropout_70/dropout/random_uniform/RandomUniform�
/sequential_46/dropout_70/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>21
/sequential_46/dropout_70/dropout/GreaterEqual/y�
-sequential_46/dropout_70/dropout/GreaterEqualGreaterEqualFsequential_46/dropout_70/dropout/random_uniform/RandomUniform:output:08sequential_46/dropout_70/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������2/
-sequential_46/dropout_70/dropout/GreaterEqual�
%sequential_46/dropout_70/dropout/CastCast1sequential_46/dropout_70/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������2'
%sequential_46/dropout_70/dropout/Cast�
&sequential_46/dropout_70/dropout/Mul_1Mul(sequential_46/dropout_70/dropout/Mul:z:0)sequential_46/dropout_70/dropout/Cast:y:0*
T0*'
_output_shapes
:���������2(
&sequential_46/dropout_70/dropout/Mul_1�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0sequential_46_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul�
IdentityIdentity*sequential_46/dropout_70/dropout/Mul_1:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp-^h_1/kernel/Regularizer/Square/ReadVariableOp)^sequential_46/h_1/BiasAdd/ReadVariableOp(^sequential_46/h_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp2T
(sequential_46/h_1/BiasAdd/ReadVariableOp(sequential_46/h_1/BiasAdd/ReadVariableOp2R
'sequential_46/h_1/MatMul/ReadVariableOp'sequential_46/h_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_output_layer_layer_call_and_return_conditional_losses_27543801

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02@
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp�
/MDN_size/output_layer/kernel/Regularizer/SquareSquareFMDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:21
/MDN_size/output_layer/kernel/Regularizer/Square�
.MDN_size/output_layer/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       20
.MDN_size/output_layer/kernel/Regularizer/Const�
,MDN_size/output_layer/kernel/Regularizer/SumSum3MDN_size/output_layer/kernel/Regularizer/Square:y:07MDN_size/output_layer/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/Sum�
.MDN_size/output_layer/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:20
.MDN_size/output_layer/kernel/Regularizer/mul/x�
,MDN_size/output_layer/kernel/Regularizer/mulMul7MDN_size/output_layer/kernel/Regularizer/mul/x:output:05MDN_size/output_layer/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,MDN_size/output_layer/kernel/Regularizer/mulm
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^BiasAdd/ReadVariableOp?^MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2�
>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp>MDN_size/output_layer/kernel/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
$__inference__traced_restore_27544291
file_prefix?
-assignvariableop_mdn_size_output_layer_kernel:;
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
assignvariableop_13_h_1_kernel:*
assignvariableop_14_h_1_bias:#
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: I
7assignvariableop_19_adam_mdn_size_output_layer_kernel_m:C
5assignvariableop_20_adam_mdn_size_output_layer_bias_m:C
1assignvariableop_21_adam_mdn_size_alphas_kernel_m:=
/assignvariableop_22_adam_mdn_size_alphas_bias_m:G
5assignvariableop_23_adam_mdn_size_distparam1_kernel_m:A
3assignvariableop_24_adam_mdn_size_distparam1_bias_m:G
5assignvariableop_25_adam_mdn_size_distparam2_kernel_m:A
3assignvariableop_26_adam_mdn_size_distparam2_bias_m:7
%assignvariableop_27_adam_h_1_kernel_m:1
#assignvariableop_28_adam_h_1_bias_m:I
7assignvariableop_29_adam_mdn_size_output_layer_kernel_v:C
5assignvariableop_30_adam_mdn_size_output_layer_bias_v:C
1assignvariableop_31_adam_mdn_size_alphas_kernel_v:=
/assignvariableop_32_adam_mdn_size_alphas_bias_v:G
5assignvariableop_33_adam_mdn_size_distparam1_kernel_v:A
3assignvariableop_34_adam_mdn_size_distparam1_bias_v:G
5assignvariableop_35_adam_mdn_size_distparam2_kernel_v:A
3assignvariableop_36_adam_mdn_size_distparam2_bias_v:7
%assignvariableop_37_adam_h_1_kernel_v:1
#assignvariableop_38_adam_h_1_bias_v:
identity_40��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*�
value�B�(B*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB(alphas/kernel/.ATTRIBUTES/VARIABLE_VALUEB&alphas/bias/.ATTRIBUTES/VARIABLE_VALUEB,distparam1/kernel/.ATTRIBUTES/VARIABLE_VALUEB*distparam1/bias/.ATTRIBUTES/VARIABLE_VALUEB,distparam2/kernel/.ATTRIBUTES/VARIABLE_VALUEB*distparam2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDalphas/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBalphas/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHdistparam1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFdistparam1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHdistparam2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFdistparam2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDalphas/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBalphas/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHdistparam1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFdistparam1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHdistparam2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFdistparam2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp-assignvariableop_mdn_size_output_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp-assignvariableop_1_mdn_size_output_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp)assignvariableop_2_mdn_size_alphas_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp'assignvariableop_3_mdn_size_alphas_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp-assignvariableop_4_mdn_size_distparam1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_mdn_size_distparam1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp-assignvariableop_6_mdn_size_distparam2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_mdn_size_distparam2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_h_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_h_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adam_mdn_size_output_layer_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_mdn_size_output_layer_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp1assignvariableop_21_adam_mdn_size_alphas_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp/assignvariableop_22_adam_mdn_size_alphas_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp5assignvariableop_23_adam_mdn_size_distparam1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp3assignvariableop_24_adam_mdn_size_distparam1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp5assignvariableop_25_adam_mdn_size_distparam2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_mdn_size_distparam2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_h_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp#assignvariableop_28_adam_h_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_mdn_size_output_layer_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_mdn_size_output_layer_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp1assignvariableop_31_adam_mdn_size_alphas_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp/assignvariableop_32_adam_mdn_size_alphas_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp5assignvariableop_33_adam_mdn_size_distparam1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_mdn_size_distparam1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_mdn_size_distparam2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_mdn_size_distparam2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp%assignvariableop_37_adam_h_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp#assignvariableop_38_adam_h_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39f
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_40�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
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
�
�
0__inference_sequential_46_layer_call_fn_27543902

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_sequential_46_layer_call_and_return_conditional_losses_275429332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_sequential_46_layer_call_and_return_conditional_losses_27542933

inputs
h_1_27542914:
h_1_27542916:
identity��h_1/StatefulPartitionedCall�,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/StatefulPartitionedCallStatefulPartitionedCallinputsh_1_27542914h_1_27542916*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_h_1_layer_call_and_return_conditional_losses_275429132
h_1/StatefulPartitionedCall�
dropout_70/PartitionedCallPartitionedCall$h_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_70_layer_call_and_return_conditional_losses_275429242
dropout_70/PartitionedCall�
,h_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOph_1_27542914*
_output_shapes

:*
dtype02.
,h_1/kernel/Regularizer/Square/ReadVariableOp�
h_1/kernel/Regularizer/SquareSquare4h_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2
h_1/kernel/Regularizer/Square�
h_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
h_1/kernel/Regularizer/Const�
h_1/kernel/Regularizer/SumSum!h_1/kernel/Regularizer/Square:y:0%h_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/Sum�
h_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
h_1/kernel/Regularizer/mul/x�
h_1/kernel/Regularizer/mulMul%h_1/kernel/Regularizer/mul/x:output:0#h_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
h_1/kernel/Regularizer/mul~
IdentityIdentity#dropout_70/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^h_1/StatefulPartitionedCall-^h_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2\
,h_1/kernel/Regularizer/Square/ReadVariableOp,h_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
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
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_model
�
nnmodel
trainable_variables
	variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
*trainable_variables
+	variables
,regularization_losses
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
.iter

/beta_1

0beta_2
	1decay
2learning_ratem�m�m�m�m�m�$m�%m�3m�4m�v�v�v�v�v�v�$v�%v�3v�4v�"
	optimizer
f
30
41
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
f
30
41
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
�
	variables
5non_trainable_variables
6metrics

7layers
8layer_regularization_losses
9layer_metrics
	regularization_losses

trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�
:layer_with_weights-0
:layer-0
;layer-1
<	variables
=regularization_losses
>trainable_variables
?	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_sequential
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
	variables
@non_trainable_variables
Ametrics
Blayer_regularization_losses
Clayer_metrics
regularization_losses

Dlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.:,2MDN_size/output_layer/kernel
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
�0"
trackable_list_wrapper
�
trainable_variables
	variables
Enon_trainable_variables
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
regularization_losses

Ilayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
trainable_variables
	variables
Jnon_trainable_variables
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
regularization_losses

Nlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
 trainable_variables
!	variables
Onon_trainable_variables
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
"regularization_losses

Slayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
&trainable_variables
'	variables
Tnon_trainable_variables
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
(regularization_losses

Xlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
*trainable_variables
+	variables
Ynon_trainable_variables
Zmetrics
[layer_regularization_losses
\layer_metrics
,regularization_losses

]layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
:2
h_1/kernel
:2h_1/bias
 "
trackable_list_wrapper
.
^0
_1"
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
�

3kernel
4bias
`trainable_variables
a	variables
bregularization_losses
c	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
.
30
41"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
�
<	variables
hnon_trainable_variables
imetrics

jlayers
klayer_regularization_losses
llayer_metrics
=regularization_losses
>trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�0"
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
	mtotal
	ncount
o	variables
p	keras_api"
_tf_keras_metric
^
	qtotal
	rcount
s
_fn_kwargs
t	variables
u	keras_api"
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
�0"
trackable_list_wrapper
�
`trainable_variables
a	variables
vnon_trainable_variables
wmetrics
xlayer_regularization_losses
ylayer_metrics
bregularization_losses

zlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
dtrainable_variables
e	variables
{non_trainable_variables
|metrics
}layer_regularization_losses
~layer_metrics
fregularization_losses

layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
m0
n1"
trackable_list_wrapper
-
o	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
-
t	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
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
3:12#Adam/MDN_size/output_layer/kernel/m
-:+2!Adam/MDN_size/output_layer/bias/m
-:+2Adam/MDN_size/alphas/kernel/m
':%2Adam/MDN_size/alphas/bias/m
1:/2!Adam/MDN_size/distparam1/kernel/m
+:)2Adam/MDN_size/distparam1/bias/m
1:/2!Adam/MDN_size/distparam2/kernel/m
+:)2Adam/MDN_size/distparam2/bias/m
!:2Adam/h_1/kernel/m
:2Adam/h_1/bias/m
3:12#Adam/MDN_size/output_layer/kernel/v
-:+2!Adam/MDN_size/output_layer/bias/v
-:+2Adam/MDN_size/alphas/kernel/v
':%2Adam/MDN_size/alphas/bias/v
1:/2!Adam/MDN_size/distparam1/kernel/v
+:)2Adam/MDN_size/distparam1/bias/v
1:/2!Adam/MDN_size/distparam2/kernel/v
+:)2Adam/MDN_size/distparam2/bias/v
!:2Adam/h_1/kernel/v
:2Adam/h_1/bias/v
�2�
+__inference_MDN_size_layer_call_fn_27543199
+__inference_MDN_size_layer_call_fn_27543562
+__inference_MDN_size_layer_call_fn_27543587
+__inference_MDN_size_layer_call_fn_27543408�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543641
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543702
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543450
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543492�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__wrapped_model_27542889input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_SeqBlock_layer_call_fn_27543717
+__inference_SeqBlock_layer_call_fn_27543726�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_SeqBlock_layer_call_and_return_conditional_losses_27543744
F__inference_SeqBlock_layer_call_and_return_conditional_losses_27543769�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_output_layer_layer_call_fn_27543784�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_output_layer_layer_call_and_return_conditional_losses_27543801�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_alphas_layer_call_fn_27543810�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_alphas_layer_call_and_return_conditional_losses_27543821�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_distparam1_layer_call_fn_27543830�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_distparam1_layer_call_and_return_conditional_losses_27543841�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_distparam2_layer_call_fn_27543850�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_distparam2_layer_call_and_return_conditional_losses_27543861�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_pvec_layer_call_fn_27543868�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_pvec_layer_call_and_return_conditional_losses_27543876�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_27543887�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
&__inference_signature_wrapper_27543537input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_sequential_46_layer_call_fn_27542940
0__inference_sequential_46_layer_call_fn_27543902
0__inference_sequential_46_layer_call_fn_27543911
0__inference_sequential_46_layer_call_fn_27543016�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543929
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543954
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543032
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543048�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_h_1_layer_call_fn_27543969�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_h_1_layer_call_and_return_conditional_losses_27543986�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_dropout_70_layer_call_fn_27543991
-__inference_dropout_70_layer_call_fn_27543996�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_dropout_70_layer_call_and_return_conditional_losses_27544001
H__inference_dropout_70_layer_call_and_return_conditional_losses_27544013�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference_loss_fn_1_27544024�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� �
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543450i
34$%4�1
*�'
!�
input_1���������
p 
� "%�"
�
0���������
� �
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543492i
34$%4�1
*�'
!�
input_1���������
p
� "%�"
�
0���������
� �
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543641h
34$%3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
F__inference_MDN_size_layer_call_and_return_conditional_losses_27543702h
34$%3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
+__inference_MDN_size_layer_call_fn_27543199\
34$%4�1
*�'
!�
input_1���������
p 
� "�����������
+__inference_MDN_size_layer_call_fn_27543408\
34$%4�1
*�'
!�
input_1���������
p
� "�����������
+__inference_MDN_size_layer_call_fn_27543562[
34$%3�0
)�&
 �
inputs���������
p 
� "�����������
+__inference_MDN_size_layer_call_fn_27543587[
34$%3�0
)�&
 �
inputs���������
p
� "�����������
F__inference_SeqBlock_layer_call_and_return_conditional_losses_27543744`343�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
F__inference_SeqBlock_layer_call_and_return_conditional_losses_27543769`343�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
+__inference_SeqBlock_layer_call_fn_27543717S343�0
)�&
 �
inputs���������
p 
� "�����������
+__inference_SeqBlock_layer_call_fn_27543726S343�0
)�&
 �
inputs���������
p
� "�����������
#__inference__wrapped_model_27542889s
34$%0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
D__inference_alphas_layer_call_and_return_conditional_losses_27543821\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� |
)__inference_alphas_layer_call_fn_27543810O/�,
%�"
 �
inputs���������
� "�����������
H__inference_distparam1_layer_call_and_return_conditional_losses_27543841\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
-__inference_distparam1_layer_call_fn_27543830O/�,
%�"
 �
inputs���������
� "�����������
H__inference_distparam2_layer_call_and_return_conditional_losses_27543861\$%/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
-__inference_distparam2_layer_call_fn_27543850O$%/�,
%�"
 �
inputs���������
� "�����������
H__inference_dropout_70_layer_call_and_return_conditional_losses_27544001\3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
H__inference_dropout_70_layer_call_and_return_conditional_losses_27544013\3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
-__inference_dropout_70_layer_call_fn_27543991O3�0
)�&
 �
inputs���������
p 
� "�����������
-__inference_dropout_70_layer_call_fn_27543996O3�0
)�&
 �
inputs���������
p
� "�����������
A__inference_h_1_layer_call_and_return_conditional_losses_27543986\34/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� y
&__inference_h_1_layer_call_fn_27543969O34/�,
%�"
 �
inputs���������
� "����������=
__inference_loss_fn_0_27543887�

� 
� "� =
__inference_loss_fn_1_275440243�

� 
� "� �
J__inference_output_layer_layer_call_and_return_conditional_losses_27543801\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
/__inference_output_layer_layer_call_fn_27543784O/�,
%�"
 �
inputs���������
� "�����������
B__inference_pvec_layer_call_and_return_conditional_losses_27543876�~�{
t�q
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
� "%�"
�
0���������
� �
'__inference_pvec_layer_call_fn_27543868�~�{
t�q
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
� "�����������
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543032g34:�7
0�-
#� 
	h_1_input���������
p 

 
� "%�"
�
0���������
� �
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543048g34:�7
0�-
#� 
	h_1_input���������
p

 
� "%�"
�
0���������
� �
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543929d347�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
K__inference_sequential_46_layer_call_and_return_conditional_losses_27543954d347�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
0__inference_sequential_46_layer_call_fn_27542940Z34:�7
0�-
#� 
	h_1_input���������
p 

 
� "�����������
0__inference_sequential_46_layer_call_fn_27543016Z34:�7
0�-
#� 
	h_1_input���������
p

 
� "�����������
0__inference_sequential_46_layer_call_fn_27543902W347�4
-�*
 �
inputs���������
p 

 
� "�����������
0__inference_sequential_46_layer_call_fn_27543911W347�4
-�*
 �
inputs���������
p

 
� "�����������
&__inference_signature_wrapper_27543537~
34$%;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������