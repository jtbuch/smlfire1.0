чз
эб
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
delete_old_dirsbool(И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
0
Sigmoid
x"T
y"T"
Ttype:

2
@
Softplus
features"T
activations"T"
Ttype:
2
Њ
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
executor_typestring И
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.02unknown8уІ

Ф
MDN_freq/output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameMDN_freq/output_layer/kernel
Н
0MDN_freq/output_layer/kernel/Read/ReadVariableOpReadVariableOpMDN_freq/output_layer/kernel*
_output_shapes

:*
dtype0
М
MDN_freq/output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameMDN_freq/output_layer/bias
Е
.MDN_freq/output_layer/bias/Read/ReadVariableOpReadVariableOpMDN_freq/output_layer/bias*
_output_shapes
:*
dtype0
А
MDN_freq/pi/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_nameMDN_freq/pi/kernel
y
&MDN_freq/pi/kernel/Read/ReadVariableOpReadVariableOpMDN_freq/pi/kernel*
_output_shapes

:*
dtype0
x
MDN_freq/pi/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameMDN_freq/pi/bias
q
$MDN_freq/pi/bias/Read/ReadVariableOpReadVariableOpMDN_freq/pi/bias*
_output_shapes
:*
dtype0
А
MDN_freq/mu/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*#
shared_nameMDN_freq/mu/kernel
y
&MDN_freq/mu/kernel/Read/ReadVariableOpReadVariableOpMDN_freq/mu/kernel*
_output_shapes

:*
dtype0
x
MDN_freq/mu/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameMDN_freq/mu/bias
q
$MDN_freq/mu/bias/Read/ReadVariableOpReadVariableOpMDN_freq/mu/bias*
_output_shapes
:*
dtype0
Ж
MDN_freq/delta/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameMDN_freq/delta/kernel

)MDN_freq/delta/kernel/Read/ReadVariableOpReadVariableOpMDN_freq/delta/kernel*
_output_shapes

:*
dtype0
~
MDN_freq/delta/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameMDN_freq/delta/bias
w
'MDN_freq/delta/bias/Read/ReadVariableOpReadVariableOpMDN_freq/delta/bias*
_output_shapes
:*
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
:*
shared_name
h_1/kernel
i
h_1/kernel/Read/ReadVariableOpReadVariableOp
h_1/kernel*
_output_shapes

:*
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
p

h_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
h_2/kernel
i
h_2/kernel/Read/ReadVariableOpReadVariableOp
h_2/kernel*
_output_shapes

:*
dtype0
h
h_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
h_2/bias
a
h_2/bias/Read/ReadVariableOpReadVariableOph_2/bias*
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
Ґ
#Adam/MDN_freq/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/MDN_freq/output_layer/kernel/m
Ы
7Adam/MDN_freq/output_layer/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/MDN_freq/output_layer/kernel/m*
_output_shapes

:*
dtype0
Ъ
!Adam/MDN_freq/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/MDN_freq/output_layer/bias/m
У
5Adam/MDN_freq/output_layer/bias/m/Read/ReadVariableOpReadVariableOp!Adam/MDN_freq/output_layer/bias/m*
_output_shapes
:*
dtype0
О
Adam/MDN_freq/pi/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameAdam/MDN_freq/pi/kernel/m
З
-Adam/MDN_freq/pi/kernel/m/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/pi/kernel/m*
_output_shapes

:*
dtype0
Ж
Adam/MDN_freq/pi/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/MDN_freq/pi/bias/m

+Adam/MDN_freq/pi/bias/m/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/pi/bias/m*
_output_shapes
:*
dtype0
О
Adam/MDN_freq/mu/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameAdam/MDN_freq/mu/kernel/m
З
-Adam/MDN_freq/mu/kernel/m/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/mu/kernel/m*
_output_shapes

:*
dtype0
Ж
Adam/MDN_freq/mu/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/MDN_freq/mu/bias/m

+Adam/MDN_freq/mu/bias/m/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/mu/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/MDN_freq/delta/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/MDN_freq/delta/kernel/m
Н
0Adam/MDN_freq/delta/kernel/m/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/delta/kernel/m*
_output_shapes

:*
dtype0
М
Adam/MDN_freq/delta/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/MDN_freq/delta/bias/m
Е
.Adam/MDN_freq/delta/bias/m/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/delta/bias/m*
_output_shapes
:*
dtype0
~
Adam/h_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/h_1/kernel/m
w
%Adam/h_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h_1/kernel/m*
_output_shapes

:*
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
~
Adam/h_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/h_2/kernel/m
w
%Adam/h_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/h_2/kernel/m*
_output_shapes

:*
dtype0
v
Adam/h_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/h_2/bias/m
o
#Adam/h_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/h_2/bias/m*
_output_shapes
:*
dtype0
Ґ
#Adam/MDN_freq/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adam/MDN_freq/output_layer/kernel/v
Ы
7Adam/MDN_freq/output_layer/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/MDN_freq/output_layer/kernel/v*
_output_shapes

:*
dtype0
Ъ
!Adam/MDN_freq/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/MDN_freq/output_layer/bias/v
У
5Adam/MDN_freq/output_layer/bias/v/Read/ReadVariableOpReadVariableOp!Adam/MDN_freq/output_layer/bias/v*
_output_shapes
:*
dtype0
О
Adam/MDN_freq/pi/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameAdam/MDN_freq/pi/kernel/v
З
-Adam/MDN_freq/pi/kernel/v/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/pi/kernel/v*
_output_shapes

:*
dtype0
Ж
Adam/MDN_freq/pi/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/MDN_freq/pi/bias/v

+Adam/MDN_freq/pi/bias/v/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/pi/bias/v*
_output_shapes
:*
dtype0
О
Adam/MDN_freq/mu/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_nameAdam/MDN_freq/mu/kernel/v
З
-Adam/MDN_freq/mu/kernel/v/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/mu/kernel/v*
_output_shapes

:*
dtype0
Ж
Adam/MDN_freq/mu/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/MDN_freq/mu/bias/v

+Adam/MDN_freq/mu/bias/v/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/mu/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/MDN_freq/delta/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/MDN_freq/delta/kernel/v
Н
0Adam/MDN_freq/delta/kernel/v/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/delta/kernel/v*
_output_shapes

:*
dtype0
М
Adam/MDN_freq/delta/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/MDN_freq/delta/bias/v
Е
.Adam/MDN_freq/delta/bias/v/Read/ReadVariableOpReadVariableOpAdam/MDN_freq/delta/bias/v*
_output_shapes
:*
dtype0
~
Adam/h_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/h_1/kernel/v
w
%Adam/h_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h_1/kernel/v*
_output_shapes

:*
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
~
Adam/h_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameAdam/h_2/kernel/v
w
%Adam/h_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/h_2/kernel/v*
_output_shapes

:*
dtype0
v
Adam/h_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/h_2/bias/v
o
#Adam/h_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/h_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
шC
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*≥C
value©CB¶C BЯC
≤
seqblock
outlayer
pi
mu
	delta
pvec
	optimizer
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
_
nnmodel
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
h

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
R
*trainable_variables
+regularization_losses
,	variables
-	keras_api
∞
.iter

/beta_1

0beta_2
	1decay
2learning_ratemВmГmДmЕmЖmЗ$mИ%mЙ3mК4mЛ5mМ6mНvОvПvРvСvТvУ$vФ%vХ3vЦ4vЧ5vШ6vЩ
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
≠
7layer_metrics
8non_trainable_variables
9metrics

:layers
trainable_variables
;layer_regularization_losses
	regularization_losses

	variables
 
†
<layer_with_weights-0
<layer-0
=layer_with_weights-1
=layer-1
>trainable_variables
?regularization_losses
@	variables
A	keras_api

30
41
52
63
 

30
41
52
63
≠
Blayer_metrics
Cnon_trainable_variables
Dmetrics

Elayers
trainable_variables
Flayer_regularization_losses
regularization_losses
	variables
\Z
VARIABLE_VALUEMDN_freq/output_layer/kernel*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEMDN_freq/output_layer/bias(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠
Glayer_metrics
Hnon_trainable_variables
Imetrics

Jlayers
trainable_variables
Klayer_regularization_losses
regularization_losses
	variables
LJ
VARIABLE_VALUEMDN_freq/pi/kernel$pi/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEMDN_freq/pi/bias"pi/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠
Llayer_metrics
Mnon_trainable_variables
Nmetrics

Olayers
trainable_variables
Player_regularization_losses
regularization_losses
	variables
LJ
VARIABLE_VALUEMDN_freq/mu/kernel$mu/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEMDN_freq/mu/bias"mu/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≠
Qlayer_metrics
Rnon_trainable_variables
Smetrics

Tlayers
 trainable_variables
Ulayer_regularization_losses
!regularization_losses
"	variables
RP
VARIABLE_VALUEMDN_freq/delta/kernel'delta/kernel/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEMDN_freq/delta/bias%delta/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
≠
Vlayer_metrics
Wnon_trainable_variables
Xmetrics

Ylayers
&trainable_variables
Zlayer_regularization_losses
'regularization_losses
(	variables
 
 
 
≠
[layer_metrics
\non_trainable_variables
]metrics

^layers
*trainable_variables
_layer_regularization_losses
+regularization_losses
,	variables
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
PN
VARIABLE_VALUE
h_1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEh_1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
h_2/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEh_2/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 

`0
a1
*
0
1
2
3
4
5
 
h

3kernel
4bias
btrainable_variables
cregularization_losses
d	variables
e	keras_api
h

5kernel
6bias
ftrainable_variables
gregularization_losses
h	variables
i	keras_api

30
41
52
63
 

30
41
52
63
≠
jlayer_metrics
knon_trainable_variables
lmetrics

mlayers
>trainable_variables
nlayer_regularization_losses
?regularization_losses
@	variables
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
 
4
	ototal
	pcount
q	variables
r	keras_api
D
	stotal
	tcount
u
_fn_kwargs
v	variables
w	keras_api

30
41
 

30
41
≠
xlayer_metrics
ynon_trainable_variables
zmetrics

{layers
btrainable_variables
|layer_regularization_losses
cregularization_losses
d	variables

50
61
 

50
61
ѓ
}layer_metrics
~non_trainable_variables
metrics
Аlayers
ftrainable_variables
 Бlayer_regularization_losses
gregularization_losses
h	variables
 
 
 

<0
=1
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

o0
p1

q	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

s0
t1

v	variables
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
VARIABLE_VALUE#Adam/MDN_freq/output_layer/kernel/mFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/MDN_freq/output_layer/bias/mDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/MDN_freq/pi/kernel/m@pi/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/MDN_freq/pi/bias/m>pi/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/MDN_freq/mu/kernel/m@mu/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/MDN_freq/mu/bias/m>mu/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/MDN_freq/delta/kernel/mCdelta/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/MDN_freq/delta/bias/mAdelta/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/h_1/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/h_1/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/h_2/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/h_2/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE#Adam/MDN_freq/output_layer/kernel/vFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!Adam/MDN_freq/output_layer/bias/vDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/MDN_freq/pi/kernel/v@pi/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/MDN_freq/pi/bias/v>pi/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/MDN_freq/mu/kernel/v@mu/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/MDN_freq/mu/bias/v>mu/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/MDN_freq/delta/kernel/vCdelta/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/MDN_freq/delta/bias/vAdelta/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/h_1/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/h_1/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/h_2/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/h_2/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Ђ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1
h_1/kernelh_1/bias
h_2/kernelh_2/biasMDN_freq/output_layer/kernelMDN_freq/output_layer/biasMDN_freq/pi/kernelMDN_freq/pi/biasMDN_freq/mu/kernelMDN_freq/mu/biasMDN_freq/delta/kernelMDN_freq/delta/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_501779
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ц
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0MDN_freq/output_layer/kernel/Read/ReadVariableOp.MDN_freq/output_layer/bias/Read/ReadVariableOp&MDN_freq/pi/kernel/Read/ReadVariableOp$MDN_freq/pi/bias/Read/ReadVariableOp&MDN_freq/mu/kernel/Read/ReadVariableOp$MDN_freq/mu/bias/Read/ReadVariableOp)MDN_freq/delta/kernel/Read/ReadVariableOp'MDN_freq/delta/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOph_1/kernel/Read/ReadVariableOph_1/bias/Read/ReadVariableOph_2/kernel/Read/ReadVariableOph_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp7Adam/MDN_freq/output_layer/kernel/m/Read/ReadVariableOp5Adam/MDN_freq/output_layer/bias/m/Read/ReadVariableOp-Adam/MDN_freq/pi/kernel/m/Read/ReadVariableOp+Adam/MDN_freq/pi/bias/m/Read/ReadVariableOp-Adam/MDN_freq/mu/kernel/m/Read/ReadVariableOp+Adam/MDN_freq/mu/bias/m/Read/ReadVariableOp0Adam/MDN_freq/delta/kernel/m/Read/ReadVariableOp.Adam/MDN_freq/delta/bias/m/Read/ReadVariableOp%Adam/h_1/kernel/m/Read/ReadVariableOp#Adam/h_1/bias/m/Read/ReadVariableOp%Adam/h_2/kernel/m/Read/ReadVariableOp#Adam/h_2/bias/m/Read/ReadVariableOp7Adam/MDN_freq/output_layer/kernel/v/Read/ReadVariableOp5Adam/MDN_freq/output_layer/bias/v/Read/ReadVariableOp-Adam/MDN_freq/pi/kernel/v/Read/ReadVariableOp+Adam/MDN_freq/pi/bias/v/Read/ReadVariableOp-Adam/MDN_freq/mu/kernel/v/Read/ReadVariableOp+Adam/MDN_freq/mu/bias/v/Read/ReadVariableOp0Adam/MDN_freq/delta/kernel/v/Read/ReadVariableOp.Adam/MDN_freq/delta/bias/v/Read/ReadVariableOp%Adam/h_1/kernel/v/Read/ReadVariableOp#Adam/h_1/bias/v/Read/ReadVariableOp%Adam/h_2/kernel/v/Read/ReadVariableOp#Adam/h_2/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_502350
н	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameMDN_freq/output_layer/kernelMDN_freq/output_layer/biasMDN_freq/pi/kernelMDN_freq/pi/biasMDN_freq/mu/kernelMDN_freq/mu/biasMDN_freq/delta/kernelMDN_freq/delta/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate
h_1/kernelh_1/bias
h_2/kernelh_2/biastotalcounttotal_1count_1#Adam/MDN_freq/output_layer/kernel/m!Adam/MDN_freq/output_layer/bias/mAdam/MDN_freq/pi/kernel/mAdam/MDN_freq/pi/bias/mAdam/MDN_freq/mu/kernel/mAdam/MDN_freq/mu/bias/mAdam/MDN_freq/delta/kernel/mAdam/MDN_freq/delta/bias/mAdam/h_1/kernel/mAdam/h_1/bias/mAdam/h_2/kernel/mAdam/h_2/bias/m#Adam/MDN_freq/output_layer/kernel/v!Adam/MDN_freq/output_layer/bias/vAdam/MDN_freq/pi/kernel/vAdam/MDN_freq/pi/bias/vAdam/MDN_freq/mu/kernel/vAdam/MDN_freq/mu/bias/vAdam/MDN_freq/delta/kernel/vAdam/MDN_freq/delta/bias/vAdam/h_1/kernel/vAdam/h_1/bias/vAdam/h_2/kernel/vAdam/h_2/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_502495ук
э

р
?__inference_h_2_layer_call_and_return_conditional_losses_501200

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ъ
•
)__inference_MDN_freq_layer_call_fn_501808

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_MDN_freq_layer_call_and_return_conditional_losses_5014332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ж
щ
H__inference_output_layer_layer_call_and_return_conditional_losses_502015

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Э
¶
)__inference_MDN_freq_layer_call_fn_501674
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_MDN_freq_layer_call_and_return_conditional_losses_5016182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
ъ
Ч
D__inference_SeqBlock_layer_call_and_return_conditional_losses_501995

inputsB
0sequential_12_h_1_matmul_readvariableop_resource:?
1sequential_12_h_1_biasadd_readvariableop_resource:B
0sequential_12_h_2_matmul_readvariableop_resource:?
1sequential_12_h_2_biasadd_readvariableop_resource:
identityИҐ(sequential_12/h_1/BiasAdd/ReadVariableOpҐ'sequential_12/h_1/MatMul/ReadVariableOpҐ(sequential_12/h_2/BiasAdd/ReadVariableOpҐ'sequential_12/h_2/MatMul/ReadVariableOp√
'sequential_12/h_1/MatMul/ReadVariableOpReadVariableOp0sequential_12_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_12/h_1/MatMul/ReadVariableOp©
sequential_12/h_1/MatMulMatMulinputs/sequential_12/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/MatMul¬
(sequential_12/h_1/BiasAdd/ReadVariableOpReadVariableOp1sequential_12_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_12/h_1/BiasAdd/ReadVariableOp…
sequential_12/h_1/BiasAddBiasAdd"sequential_12/h_1/MatMul:product:00sequential_12/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/BiasAddО
sequential_12/h_1/ReluRelu"sequential_12/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/Relu√
'sequential_12/h_2/MatMul/ReadVariableOpReadVariableOp0sequential_12_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_12/h_2/MatMul/ReadVariableOp«
sequential_12/h_2/MatMulMatMul$sequential_12/h_1/Relu:activations:0/sequential_12/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/MatMul¬
(sequential_12/h_2/BiasAdd/ReadVariableOpReadVariableOp1sequential_12_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_12/h_2/BiasAdd/ReadVariableOp…
sequential_12/h_2/BiasAddBiasAdd"sequential_12/h_2/MatMul:product:00sequential_12/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/BiasAddО
sequential_12/h_2/ReluRelu"sequential_12/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/Relu
IdentityIdentity$sequential_12/h_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityш
NoOpNoOp)^sequential_12/h_1/BiasAdd/ReadVariableOp(^sequential_12/h_1/MatMul/ReadVariableOp)^sequential_12/h_2/BiasAdd/ReadVariableOp(^sequential_12/h_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2T
(sequential_12/h_1/BiasAdd/ReadVariableOp(sequential_12/h_1/BiasAdd/ReadVariableOp2R
'sequential_12/h_1/MatMul/ReadVariableOp'sequential_12/h_1/MatMul/ReadVariableOp2T
(sequential_12/h_2/BiasAdd/ReadVariableOp(sequential_12/h_2/BiasAdd/ReadVariableOp2R
'sequential_12/h_2/MatMul/ReadVariableOp'sequential_12/h_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ю

п
>__inference_pi_layer_call_and_return_conditional_losses_501382

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Р
Ч
I__inference_sequential_12_layer_call_and_return_conditional_losses_501305
	h_1_input

h_1_501294:

h_1_501296:

h_2_501299:

h_2_501301:
identityИҐh_1/StatefulPartitionedCallҐh_2/StatefulPartitionedCallБ
h_1/StatefulPartitionedCallStatefulPartitionedCall	h_1_input
h_1_501294
h_1_501296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_h_1_layer_call_and_return_conditional_losses_5011832
h_1/StatefulPartitionedCallЬ
h_2/StatefulPartitionedCallStatefulPartitionedCall$h_1/StatefulPartitionedCall:output:0
h_2_501299
h_2_501301*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_h_2_layer_call_and_return_conditional_losses_5012002
h_2/StatefulPartitionedCall
IdentityIdentity$h_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityК
NoOpNoOp^h_1/StatefulPartitionedCall^h_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2:
h_2/StatefulPartitionedCallh_2/StatefulPartitionedCall:R N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	h_1_input
Ё
z
@__inference_pvec_layer_call_and_return_conditional_losses_502090
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЛ
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2
э

р
?__inference_h_1_layer_call_and_return_conditional_losses_501183

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
”
x
@__inference_pvec_layer_call_and_return_conditional_losses_501430

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЙ
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ъ
•
)__inference_MDN_freq_layer_call_fn_501837

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_MDN_freq_layer_call_and_return_conditional_losses_5016182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
к
С
$__inference_h_1_layer_call_fn_502161

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_h_1_layer_call_and_return_conditional_losses_5011832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ь
Ъ
-__inference_output_layer_layer_call_fn_502004

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_5013652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ж
щ
H__inference_output_layer_layer_call_and_return_conditional_losses_501365

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Э
¶
)__inference_MDN_freq_layer_call_fn_501460
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_MDN_freq_layer_call_and_return_conditional_losses_5014332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
‘ 
э
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501433

inputs!
seqblock_501345:
seqblock_501347:!
seqblock_501349:
seqblock_501351:%
output_layer_501366:!
output_layer_501368:
	pi_501383:
	pi_501385:
	mu_501400:
	mu_501402:
delta_501417:
delta_501419:
identityИҐ SeqBlock/StatefulPartitionedCallҐdelta/StatefulPartitionedCallҐmu/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallҐpi/StatefulPartitionedCallљ
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinputsseqblock_501345seqblock_501347seqblock_501349seqblock_501351*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_SeqBlock_layer_call_and_return_conditional_losses_5013442"
 SeqBlock/StatefulPartitionedCallќ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_501366output_layer_501368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_5013652&
$output_layer/StatefulPartitionedCall†
pi/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0	pi_501383	pi_501385*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_pi_layer_call_and_return_conditional_losses_5013822
pi/StatefulPartitionedCall†
mu/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0	mu_501400	mu_501402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_mu_layer_call_and_return_conditional_losses_5013992
mu/StatefulPartitionedCallѓ
delta/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0delta_501417delta_501419*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_delta_layer_call_and_return_conditional_losses_5014162
delta/StatefulPartitionedCallЈ
pvec/PartitionedCallPartitionedCall#pi/StatefulPartitionedCall:output:0#mu/StatefulPartitionedCall:output:0&delta/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_pvec_layer_call_and_return_conditional_losses_5014302
pvec/PartitionedCallx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityт
NoOpNoOp!^SeqBlock/StatefulPartitionedCall^delta/StatefulPartitionedCall^mu/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall^pi/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2>
delta/StatefulPartitionedCalldelta/StatefulPartitionedCall28
mu/StatefulPartitionedCallmu/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
и
Р
#__inference_mu_layer_call_fn_502044

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_mu_layer_call_and_return_conditional_losses_5013992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
„ 
ю
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501708
input_1!
seqblock_501677:
seqblock_501679:!
seqblock_501681:
seqblock_501683:%
output_layer_501686:!
output_layer_501688:
	pi_501691:
	pi_501693:
	mu_501696:
	mu_501698:
delta_501701:
delta_501703:
identityИҐ SeqBlock/StatefulPartitionedCallҐdelta/StatefulPartitionedCallҐmu/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallҐpi/StatefulPartitionedCallЊ
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinput_1seqblock_501677seqblock_501679seqblock_501681seqblock_501683*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_SeqBlock_layer_call_and_return_conditional_losses_5013442"
 SeqBlock/StatefulPartitionedCallќ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_501686output_layer_501688*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_5013652&
$output_layer/StatefulPartitionedCall†
pi/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0	pi_501691	pi_501693*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_pi_layer_call_and_return_conditional_losses_5013822
pi/StatefulPartitionedCall†
mu/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0	mu_501696	mu_501698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_mu_layer_call_and_return_conditional_losses_5013992
mu/StatefulPartitionedCallѓ
delta/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0delta_501701delta_501703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_delta_layer_call_and_return_conditional_losses_5014162
delta/StatefulPartitionedCallЈ
pvec/PartitionedCallPartitionedCall#pi/StatefulPartitionedCall:output:0#mu/StatefulPartitionedCall:output:0&delta/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_pvec_layer_call_and_return_conditional_losses_5014302
pvec/PartitionedCallx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityт
NoOpNoOp!^SeqBlock/StatefulPartitionedCall^delta/StatefulPartitionedCall^mu/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall^pi/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2>
delta/StatefulPartitionedCalldelta/StatefulPartitionedCall28
mu/StatefulPartitionedCallmu/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
џ
‘
.__inference_sequential_12_layer_call_fn_501218
	h_1_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCall	h_1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_5012072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	h_1_input
ъ
Ч
D__inference_SeqBlock_layer_call_and_return_conditional_losses_501344

inputsB
0sequential_12_h_1_matmul_readvariableop_resource:?
1sequential_12_h_1_biasadd_readvariableop_resource:B
0sequential_12_h_2_matmul_readvariableop_resource:?
1sequential_12_h_2_biasadd_readvariableop_resource:
identityИҐ(sequential_12/h_1/BiasAdd/ReadVariableOpҐ'sequential_12/h_1/MatMul/ReadVariableOpҐ(sequential_12/h_2/BiasAdd/ReadVariableOpҐ'sequential_12/h_2/MatMul/ReadVariableOp√
'sequential_12/h_1/MatMul/ReadVariableOpReadVariableOp0sequential_12_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_12/h_1/MatMul/ReadVariableOp©
sequential_12/h_1/MatMulMatMulinputs/sequential_12/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/MatMul¬
(sequential_12/h_1/BiasAdd/ReadVariableOpReadVariableOp1sequential_12_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_12/h_1/BiasAdd/ReadVariableOp…
sequential_12/h_1/BiasAddBiasAdd"sequential_12/h_1/MatMul:product:00sequential_12/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/BiasAddО
sequential_12/h_1/ReluRelu"sequential_12/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/Relu√
'sequential_12/h_2/MatMul/ReadVariableOpReadVariableOp0sequential_12_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_12/h_2/MatMul/ReadVariableOp«
sequential_12/h_2/MatMulMatMul$sequential_12/h_1/Relu:activations:0/sequential_12/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/MatMul¬
(sequential_12/h_2/BiasAdd/ReadVariableOpReadVariableOp1sequential_12_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_12/h_2/BiasAdd/ReadVariableOp…
sequential_12/h_2/BiasAddBiasAdd"sequential_12/h_2/MatMul:product:00sequential_12/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/BiasAddО
sequential_12/h_2/ReluRelu"sequential_12/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/Relu
IdentityIdentity$sequential_12/h_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityш
NoOpNoOp)^sequential_12/h_1/BiasAdd/ReadVariableOp(^sequential_12/h_1/MatMul/ReadVariableOp)^sequential_12/h_2/BiasAdd/ReadVariableOp(^sequential_12/h_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2T
(sequential_12/h_1/BiasAdd/ReadVariableOp(sequential_12/h_1/BiasAdd/ReadVariableOp2R
'sequential_12/h_1/MatMul/ReadVariableOp'sequential_12/h_1/MatMul/ReadVariableOp2T
(sequential_12/h_2/BiasAdd/ReadVariableOp(sequential_12/h_2/BiasAdd/ReadVariableOp2R
'sequential_12/h_2/MatMul/ReadVariableOp'sequential_12/h_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ДC
Л

D__inference_MDN_freq_layer_call_and_return_conditional_losses_501933

inputsK
9seqblock_sequential_12_h_1_matmul_readvariableop_resource:H
:seqblock_sequential_12_h_1_biasadd_readvariableop_resource:K
9seqblock_sequential_12_h_2_matmul_readvariableop_resource:H
:seqblock_sequential_12_h_2_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:3
!pi_matmul_readvariableop_resource:0
"pi_biasadd_readvariableop_resource:3
!mu_matmul_readvariableop_resource:0
"mu_biasadd_readvariableop_resource:6
$delta_matmul_readvariableop_resource:3
%delta_biasadd_readvariableop_resource:
identityИҐ1SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOpҐ0SeqBlock/sequential_12/h_1/MatMul/ReadVariableOpҐ1SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOpҐ0SeqBlock/sequential_12/h_2/MatMul/ReadVariableOpҐdelta/BiasAdd/ReadVariableOpҐdelta/MatMul/ReadVariableOpҐmu/BiasAdd/ReadVariableOpҐmu/MatMul/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOpҐpi/BiasAdd/ReadVariableOpҐpi/MatMul/ReadVariableOpё
0SeqBlock/sequential_12/h_1/MatMul/ReadVariableOpReadVariableOp9seqblock_sequential_12_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype022
0SeqBlock/sequential_12/h_1/MatMul/ReadVariableOpƒ
!SeqBlock/sequential_12/h_1/MatMulMatMulinputs8SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2#
!SeqBlock/sequential_12/h_1/MatMulЁ
1SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOpReadVariableOp:seqblock_sequential_12_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOpн
"SeqBlock/sequential_12/h_1/BiasAddBiasAdd+SeqBlock/sequential_12/h_1/MatMul:product:09SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2$
"SeqBlock/sequential_12/h_1/BiasAdd©
SeqBlock/sequential_12/h_1/ReluRelu+SeqBlock/sequential_12/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2!
SeqBlock/sequential_12/h_1/Reluё
0SeqBlock/sequential_12/h_2/MatMul/ReadVariableOpReadVariableOp9seqblock_sequential_12_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype022
0SeqBlock/sequential_12/h_2/MatMul/ReadVariableOpл
!SeqBlock/sequential_12/h_2/MatMulMatMul-SeqBlock/sequential_12/h_1/Relu:activations:08SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2#
!SeqBlock/sequential_12/h_2/MatMulЁ
1SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOpReadVariableOp:seqblock_sequential_12_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOpн
"SeqBlock/sequential_12/h_2/BiasAddBiasAdd+SeqBlock/sequential_12/h_2/MatMul:product:09SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2$
"SeqBlock/sequential_12/h_2/BiasAdd©
SeqBlock/sequential_12/h_2/ReluRelu+SeqBlock/sequential_12/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2!
SeqBlock/sequential_12/h_2/Reluі
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"output_layer/MatMul/ReadVariableOpЅ
output_layer/MatMulMatMul-SeqBlock/sequential_12/h_2/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/MatMul≥
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOpµ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/BiasAdd
output_layer/ReluReluoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/ReluЦ
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
pi/MatMul/ReadVariableOpХ
	pi/MatMulMatMuloutput_layer/Relu:activations:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
	pi/MatMulХ
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pi/BiasAdd/ReadVariableOpН

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

pi/BiasAddj

pi/SigmoidSigmoidpi/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

pi/SigmoidЦ
mu/MatMul/ReadVariableOpReadVariableOp!mu_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
mu/MatMul/ReadVariableOpХ
	mu/MatMulMatMuloutput_layer/Relu:activations:0 mu/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
	mu/MatMulХ
mu/BiasAdd/ReadVariableOpReadVariableOp"mu_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mu/BiasAdd/ReadVariableOpН

mu/BiasAddBiasAddmu/MatMul:product:0!mu/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

mu/BiasAddm
mu/SoftplusSoftplusmu/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
mu/SoftplusЯ
delta/MatMul/ReadVariableOpReadVariableOp$delta_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
delta/MatMul/ReadVariableOpЮ
delta/MatMulMatMuloutput_layer/Relu:activations:0#delta/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
delta/MatMulЮ
delta/BiasAdd/ReadVariableOpReadVariableOp%delta_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
delta/BiasAdd/ReadVariableOpЩ
delta/BiasAddBiasAdddelta/MatMul:product:0$delta/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
delta/BiasAddv
delta/SoftplusSoftplusdelta/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
delta/Softplusf
pvec/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
pvec/concat/axis≈
pvec/concatConcatV2pi/Sigmoid:y:0mu/Softplus:activations:0delta/Softplus:activations:0pvec/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€2
pvec/concato
IdentityIdentitypvec/concat:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityТ
NoOpNoOp2^SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp1^SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp2^SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp1^SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp^delta/BiasAdd/ReadVariableOp^delta/MatMul/ReadVariableOp^mu/BiasAdd/ReadVariableOp^mu/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp^pi/BiasAdd/ReadVariableOp^pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 2f
1SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp1SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp2d
0SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp0SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp2f
1SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp1SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp2d
0SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp0SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp2<
delta/BiasAdd/ReadVariableOpdelta/BiasAdd/ReadVariableOp2:
delta/MatMul/ReadVariableOpdelta/MatMul/ReadVariableOp26
mu/BiasAdd/ReadVariableOpmu/BiasAdd/ReadVariableOp24
mu/MatMul/ReadVariableOpmu/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp26
pi/BiasAdd/ReadVariableOppi/BiasAdd/ReadVariableOp24
pi/MatMul/ReadVariableOppi/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
И
ђ
I__inference_sequential_12_layer_call_and_return_conditional_losses_502134

inputs4
"h_1_matmul_readvariableop_resource:1
#h_1_biasadd_readvariableop_resource:4
"h_2_matmul_readvariableop_resource:1
#h_2_biasadd_readvariableop_resource:
identityИҐh_1/BiasAdd/ReadVariableOpҐh_1/MatMul/ReadVariableOpҐh_2/BiasAdd/ReadVariableOpҐh_2/MatMul/ReadVariableOpЩ
h_1/MatMul/ReadVariableOpReadVariableOp"h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h_1/MatMul/ReadVariableOp

h_1/MatMulMatMulinputs!h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

h_1/MatMulШ
h_1/BiasAdd/ReadVariableOpReadVariableOp#h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h_1/BiasAdd/ReadVariableOpС
h_1/BiasAddBiasAddh_1/MatMul:product:0"h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
h_1/BiasAddd
h_1/ReluReluh_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

h_1/ReluЩ
h_2/MatMul/ReadVariableOpReadVariableOp"h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h_2/MatMul/ReadVariableOpП

h_2/MatMulMatMulh_1/Relu:activations:0!h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

h_2/MatMulШ
h_2/BiasAdd/ReadVariableOpReadVariableOp#h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h_2/BiasAdd/ReadVariableOpС
h_2/BiasAddBiasAddh_2/MatMul:product:0"h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
h_2/BiasAddd
h_2/ReluReluh_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

h_2/Reluq
IdentityIdentityh_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityј
NoOpNoOp^h_1/BiasAdd/ReadVariableOp^h_1/MatMul/ReadVariableOp^h_2/BiasAdd/ReadVariableOp^h_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 28
h_1/BiasAdd/ReadVariableOph_1/BiasAdd/ReadVariableOp26
h_1/MatMul/ReadVariableOph_1/MatMul/ReadVariableOp28
h_2/BiasAdd/ReadVariableOph_2/BiasAdd/ReadVariableOp26
h_2/MatMul/ReadVariableOph_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
“
—
.__inference_sequential_12_layer_call_fn_502116

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_5012672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
З
Ф
I__inference_sequential_12_layer_call_and_return_conditional_losses_501267

inputs

h_1_501256:

h_1_501258:

h_2_501261:

h_2_501263:
identityИҐh_1/StatefulPartitionedCallҐh_2/StatefulPartitionedCallю
h_1/StatefulPartitionedCallStatefulPartitionedCallinputs
h_1_501256
h_1_501258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_h_1_layer_call_and_return_conditional_losses_5011832
h_1/StatefulPartitionedCallЬ
h_2/StatefulPartitionedCallStatefulPartitionedCall$h_1/StatefulPartitionedCall:output:0
h_2_501261
h_2_501263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_h_2_layer_call_and_return_conditional_losses_5012002
h_2/StatefulPartitionedCall
IdentityIdentity$h_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityК
NoOpNoOp^h_1/StatefulPartitionedCall^h_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2:
h_2/StatefulPartitionedCallh_2/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ДC
Л

D__inference_MDN_freq_layer_call_and_return_conditional_losses_501885

inputsK
9seqblock_sequential_12_h_1_matmul_readvariableop_resource:H
:seqblock_sequential_12_h_1_biasadd_readvariableop_resource:K
9seqblock_sequential_12_h_2_matmul_readvariableop_resource:H
:seqblock_sequential_12_h_2_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:3
!pi_matmul_readvariableop_resource:0
"pi_biasadd_readvariableop_resource:3
!mu_matmul_readvariableop_resource:0
"mu_biasadd_readvariableop_resource:6
$delta_matmul_readvariableop_resource:3
%delta_biasadd_readvariableop_resource:
identityИҐ1SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOpҐ0SeqBlock/sequential_12/h_1/MatMul/ReadVariableOpҐ1SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOpҐ0SeqBlock/sequential_12/h_2/MatMul/ReadVariableOpҐdelta/BiasAdd/ReadVariableOpҐdelta/MatMul/ReadVariableOpҐmu/BiasAdd/ReadVariableOpҐmu/MatMul/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOpҐpi/BiasAdd/ReadVariableOpҐpi/MatMul/ReadVariableOpё
0SeqBlock/sequential_12/h_1/MatMul/ReadVariableOpReadVariableOp9seqblock_sequential_12_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype022
0SeqBlock/sequential_12/h_1/MatMul/ReadVariableOpƒ
!SeqBlock/sequential_12/h_1/MatMulMatMulinputs8SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2#
!SeqBlock/sequential_12/h_1/MatMulЁ
1SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOpReadVariableOp:seqblock_sequential_12_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOpн
"SeqBlock/sequential_12/h_1/BiasAddBiasAdd+SeqBlock/sequential_12/h_1/MatMul:product:09SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2$
"SeqBlock/sequential_12/h_1/BiasAdd©
SeqBlock/sequential_12/h_1/ReluRelu+SeqBlock/sequential_12/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2!
SeqBlock/sequential_12/h_1/Reluё
0SeqBlock/sequential_12/h_2/MatMul/ReadVariableOpReadVariableOp9seqblock_sequential_12_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype022
0SeqBlock/sequential_12/h_2/MatMul/ReadVariableOpл
!SeqBlock/sequential_12/h_2/MatMulMatMul-SeqBlock/sequential_12/h_1/Relu:activations:08SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2#
!SeqBlock/sequential_12/h_2/MatMulЁ
1SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOpReadVariableOp:seqblock_sequential_12_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOpн
"SeqBlock/sequential_12/h_2/BiasAddBiasAdd+SeqBlock/sequential_12/h_2/MatMul:product:09SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2$
"SeqBlock/sequential_12/h_2/BiasAdd©
SeqBlock/sequential_12/h_2/ReluRelu+SeqBlock/sequential_12/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2!
SeqBlock/sequential_12/h_2/Reluі
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"output_layer/MatMul/ReadVariableOpЅ
output_layer/MatMulMatMul-SeqBlock/sequential_12/h_2/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/MatMul≥
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOpµ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/BiasAdd
output_layer/ReluReluoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/ReluЦ
pi/MatMul/ReadVariableOpReadVariableOp!pi_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
pi/MatMul/ReadVariableOpХ
	pi/MatMulMatMuloutput_layer/Relu:activations:0 pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
	pi/MatMulХ
pi/BiasAdd/ReadVariableOpReadVariableOp"pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
pi/BiasAdd/ReadVariableOpН

pi/BiasAddBiasAddpi/MatMul:product:0!pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

pi/BiasAddj

pi/SigmoidSigmoidpi/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

pi/SigmoidЦ
mu/MatMul/ReadVariableOpReadVariableOp!mu_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
mu/MatMul/ReadVariableOpХ
	mu/MatMulMatMuloutput_layer/Relu:activations:0 mu/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
	mu/MatMulХ
mu/BiasAdd/ReadVariableOpReadVariableOp"mu_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
mu/BiasAdd/ReadVariableOpН

mu/BiasAddBiasAddmu/MatMul:product:0!mu/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

mu/BiasAddm
mu/SoftplusSoftplusmu/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
mu/SoftplusЯ
delta/MatMul/ReadVariableOpReadVariableOp$delta_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
delta/MatMul/ReadVariableOpЮ
delta/MatMulMatMuloutput_layer/Relu:activations:0#delta/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
delta/MatMulЮ
delta/BiasAdd/ReadVariableOpReadVariableOp%delta_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
delta/BiasAdd/ReadVariableOpЩ
delta/BiasAddBiasAdddelta/MatMul:product:0$delta/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
delta/BiasAddv
delta/SoftplusSoftplusdelta/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
delta/Softplusf
pvec/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
pvec/concat/axis≈
pvec/concatConcatV2pi/Sigmoid:y:0mu/Softplus:activations:0delta/Softplus:activations:0pvec/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€2
pvec/concato
IdentityIdentitypvec/concat:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityТ
NoOpNoOp2^SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp1^SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp2^SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp1^SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp^delta/BiasAdd/ReadVariableOp^delta/MatMul/ReadVariableOp^mu/BiasAdd/ReadVariableOp^mu/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp^pi/BiasAdd/ReadVariableOp^pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 2f
1SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp1SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp2d
0SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp0SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp2f
1SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp1SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp2d
0SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp0SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp2<
delta/BiasAdd/ReadVariableOpdelta/BiasAdd/ReadVariableOp2:
delta/MatMul/ReadVariableOpdelta/MatMul/ReadVariableOp26
mu/BiasAdd/ReadVariableOpmu/BiasAdd/ReadVariableOp24
mu/MatMul/ReadVariableOpmu/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp26
pi/BiasAdd/ReadVariableOppi/BiasAdd/ReadVariableOp24
pi/MatMul/ReadVariableOppi/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
о
У
&__inference_delta_layer_call_fn_502064

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_delta_layer_call_and_return_conditional_losses_5014162
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
П
т
A__inference_delta_layer_call_and_return_conditional_losses_501416

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ъ
Ч
D__inference_SeqBlock_layer_call_and_return_conditional_losses_501542

inputsB
0sequential_12_h_1_matmul_readvariableop_resource:?
1sequential_12_h_1_biasadd_readvariableop_resource:B
0sequential_12_h_2_matmul_readvariableop_resource:?
1sequential_12_h_2_biasadd_readvariableop_resource:
identityИҐ(sequential_12/h_1/BiasAdd/ReadVariableOpҐ'sequential_12/h_1/MatMul/ReadVariableOpҐ(sequential_12/h_2/BiasAdd/ReadVariableOpҐ'sequential_12/h_2/MatMul/ReadVariableOp√
'sequential_12/h_1/MatMul/ReadVariableOpReadVariableOp0sequential_12_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_12/h_1/MatMul/ReadVariableOp©
sequential_12/h_1/MatMulMatMulinputs/sequential_12/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/MatMul¬
(sequential_12/h_1/BiasAdd/ReadVariableOpReadVariableOp1sequential_12_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_12/h_1/BiasAdd/ReadVariableOp…
sequential_12/h_1/BiasAddBiasAdd"sequential_12/h_1/MatMul:product:00sequential_12/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/BiasAddО
sequential_12/h_1/ReluRelu"sequential_12/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/Relu√
'sequential_12/h_2/MatMul/ReadVariableOpReadVariableOp0sequential_12_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_12/h_2/MatMul/ReadVariableOp«
sequential_12/h_2/MatMulMatMul$sequential_12/h_1/Relu:activations:0/sequential_12/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/MatMul¬
(sequential_12/h_2/BiasAdd/ReadVariableOpReadVariableOp1sequential_12_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_12/h_2/BiasAdd/ReadVariableOp…
sequential_12/h_2/BiasAddBiasAdd"sequential_12/h_2/MatMul:product:00sequential_12/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/BiasAddО
sequential_12/h_2/ReluRelu"sequential_12/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/Relu
IdentityIdentity$sequential_12/h_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityш
NoOpNoOp)^sequential_12/h_1/BiasAdd/ReadVariableOp(^sequential_12/h_1/MatMul/ReadVariableOp)^sequential_12/h_2/BiasAdd/ReadVariableOp(^sequential_12/h_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2T
(sequential_12/h_1/BiasAdd/ReadVariableOp(sequential_12/h_1/BiasAdd/ReadVariableOp2R
'sequential_12/h_1/MatMul/ReadVariableOp'sequential_12/h_1/MatMul/ReadVariableOp2T
(sequential_12/h_2/BiasAdd/ReadVariableOp(sequential_12/h_2/BiasAdd/ReadVariableOp2R
'sequential_12/h_2/MatMul/ReadVariableOp'sequential_12/h_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Р
Ч
I__inference_sequential_12_layer_call_and_return_conditional_losses_501319
	h_1_input

h_1_501308:

h_1_501310:

h_2_501313:

h_2_501315:
identityИҐh_1/StatefulPartitionedCallҐh_2/StatefulPartitionedCallБ
h_1/StatefulPartitionedCallStatefulPartitionedCall	h_1_input
h_1_501308
h_1_501310*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_h_1_layer_call_and_return_conditional_losses_5011832
h_1/StatefulPartitionedCallЬ
h_2/StatefulPartitionedCallStatefulPartitionedCall$h_1/StatefulPartitionedCall:output:0
h_2_501313
h_2_501315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_h_2_layer_call_and_return_conditional_losses_5012002
h_2/StatefulPartitionedCall
IdentityIdentity$h_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityК
NoOpNoOp^h_1/StatefulPartitionedCall^h_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2:
h_2/StatefulPartitionedCallh_2/StatefulPartitionedCall:R N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	h_1_input
‘ 
э
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501618

inputs!
seqblock_501587:
seqblock_501589:!
seqblock_501591:
seqblock_501593:%
output_layer_501596:!
output_layer_501598:
	pi_501601:
	pi_501603:
	mu_501606:
	mu_501608:
delta_501611:
delta_501613:
identityИҐ SeqBlock/StatefulPartitionedCallҐdelta/StatefulPartitionedCallҐmu/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallҐpi/StatefulPartitionedCallљ
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinputsseqblock_501587seqblock_501589seqblock_501591seqblock_501593*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_SeqBlock_layer_call_and_return_conditional_losses_5015422"
 SeqBlock/StatefulPartitionedCallќ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_501596output_layer_501598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_5013652&
$output_layer/StatefulPartitionedCall†
pi/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0	pi_501601	pi_501603*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_pi_layer_call_and_return_conditional_losses_5013822
pi/StatefulPartitionedCall†
mu/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0	mu_501606	mu_501608*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_mu_layer_call_and_return_conditional_losses_5013992
mu/StatefulPartitionedCallѓ
delta/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0delta_501611delta_501613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_delta_layer_call_and_return_conditional_losses_5014162
delta/StatefulPartitionedCallЈ
pvec/PartitionedCallPartitionedCall#pi/StatefulPartitionedCall:output:0#mu/StatefulPartitionedCall:output:0&delta/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_pvec_layer_call_and_return_conditional_losses_5014302
pvec/PartitionedCallx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityт
NoOpNoOp!^SeqBlock/StatefulPartitionedCall^delta/StatefulPartitionedCall^mu/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall^pi/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2>
delta/StatefulPartitionedCalldelta/StatefulPartitionedCall28
mu/StatefulPartitionedCallmu/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»
ћ
)__inference_SeqBlock_layer_call_fn_501946

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_SeqBlock_layer_call_and_return_conditional_losses_5013442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»
ћ
)__inference_SeqBlock_layer_call_fn_501959

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_SeqBlock_layer_call_and_return_conditional_losses_5015422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
„ 
ю
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501742
input_1!
seqblock_501711:
seqblock_501713:!
seqblock_501715:
seqblock_501717:%
output_layer_501720:!
output_layer_501722:
	pi_501725:
	pi_501727:
	mu_501730:
	mu_501732:
delta_501735:
delta_501737:
identityИҐ SeqBlock/StatefulPartitionedCallҐdelta/StatefulPartitionedCallҐmu/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallҐpi/StatefulPartitionedCallЊ
 SeqBlock/StatefulPartitionedCallStatefulPartitionedCallinput_1seqblock_501711seqblock_501713seqblock_501715seqblock_501717*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_SeqBlock_layer_call_and_return_conditional_losses_5015422"
 SeqBlock/StatefulPartitionedCallќ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall)SeqBlock/StatefulPartitionedCall:output:0output_layer_501720output_layer_501722*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_5013652&
$output_layer/StatefulPartitionedCall†
pi/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0	pi_501725	pi_501727*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_pi_layer_call_and_return_conditional_losses_5013822
pi/StatefulPartitionedCall†
mu/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0	mu_501730	mu_501732*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_mu_layer_call_and_return_conditional_losses_5013992
mu/StatefulPartitionedCallѓ
delta/StatefulPartitionedCallStatefulPartitionedCall-output_layer/StatefulPartitionedCall:output:0delta_501735delta_501737*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_delta_layer_call_and_return_conditional_losses_5014162
delta/StatefulPartitionedCallЈ
pvec/PartitionedCallPartitionedCall#pi/StatefulPartitionedCall:output:0#mu/StatefulPartitionedCall:output:0&delta/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_pvec_layer_call_and_return_conditional_losses_5014302
pvec/PartitionedCallx
IdentityIdentitypvec/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityт
NoOpNoOp!^SeqBlock/StatefulPartitionedCall^delta/StatefulPartitionedCall^mu/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall^pi/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 2D
 SeqBlock/StatefulPartitionedCall SeqBlock/StatefulPartitionedCall2>
delta/StatefulPartitionedCalldelta/StatefulPartitionedCall28
mu/StatefulPartitionedCallmu/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall28
pi/StatefulPartitionedCallpi/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
И
ђ
I__inference_sequential_12_layer_call_and_return_conditional_losses_502152

inputs4
"h_1_matmul_readvariableop_resource:1
#h_1_biasadd_readvariableop_resource:4
"h_2_matmul_readvariableop_resource:1
#h_2_biasadd_readvariableop_resource:
identityИҐh_1/BiasAdd/ReadVariableOpҐh_1/MatMul/ReadVariableOpҐh_2/BiasAdd/ReadVariableOpҐh_2/MatMul/ReadVariableOpЩ
h_1/MatMul/ReadVariableOpReadVariableOp"h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h_1/MatMul/ReadVariableOp

h_1/MatMulMatMulinputs!h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

h_1/MatMulШ
h_1/BiasAdd/ReadVariableOpReadVariableOp#h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h_1/BiasAdd/ReadVariableOpС
h_1/BiasAddBiasAddh_1/MatMul:product:0"h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
h_1/BiasAddd
h_1/ReluReluh_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

h_1/ReluЩ
h_2/MatMul/ReadVariableOpReadVariableOp"h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
h_2/MatMul/ReadVariableOpП

h_2/MatMulMatMulh_1/Relu:activations:0!h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

h_2/MatMulШ
h_2/BiasAdd/ReadVariableOpReadVariableOp#h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
h_2/BiasAdd/ReadVariableOpС
h_2/BiasAddBiasAddh_2/MatMul:product:0"h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
h_2/BiasAddd
h_2/ReluReluh_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

h_2/Reluq
IdentityIdentityh_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityј
NoOpNoOp^h_1/BiasAdd/ReadVariableOp^h_1/MatMul/ReadVariableOp^h_2/BiasAdd/ReadVariableOp^h_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 28
h_1/BiasAdd/ReadVariableOph_1/BiasAdd/ReadVariableOp26
h_1/MatMul/ReadVariableOph_1/MatMul/ReadVariableOp28
h_2/BiasAdd/ReadVariableOph_2/BiasAdd/ReadVariableOp26
h_2/MatMul/ReadVariableOph_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ї
_
%__inference_pvec_layer_call_fn_502082
inputs_0
inputs_1
inputs_2
identityў
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_pvec_layer_call_and_return_conditional_losses_5014302
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/2
М
п
>__inference_mu_layer_call_and_return_conditional_losses_501399

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
и
Р
#__inference_pi_layer_call_fn_502024

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_pi_layer_call_and_return_conditional_losses_5013822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
э

р
?__inference_h_2_layer_call_and_return_conditional_losses_502192

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
к
С
$__inference_h_2_layer_call_fn_502181

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_h_2_layer_call_and_return_conditional_losses_5012002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ъ
Ч
D__inference_SeqBlock_layer_call_and_return_conditional_losses_501977

inputsB
0sequential_12_h_1_matmul_readvariableop_resource:?
1sequential_12_h_1_biasadd_readvariableop_resource:B
0sequential_12_h_2_matmul_readvariableop_resource:?
1sequential_12_h_2_biasadd_readvariableop_resource:
identityИҐ(sequential_12/h_1/BiasAdd/ReadVariableOpҐ'sequential_12/h_1/MatMul/ReadVariableOpҐ(sequential_12/h_2/BiasAdd/ReadVariableOpҐ'sequential_12/h_2/MatMul/ReadVariableOp√
'sequential_12/h_1/MatMul/ReadVariableOpReadVariableOp0sequential_12_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_12/h_1/MatMul/ReadVariableOp©
sequential_12/h_1/MatMulMatMulinputs/sequential_12/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/MatMul¬
(sequential_12/h_1/BiasAdd/ReadVariableOpReadVariableOp1sequential_12_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_12/h_1/BiasAdd/ReadVariableOp…
sequential_12/h_1/BiasAddBiasAdd"sequential_12/h_1/MatMul:product:00sequential_12/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/BiasAddО
sequential_12/h_1/ReluRelu"sequential_12/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_1/Relu√
'sequential_12/h_2/MatMul/ReadVariableOpReadVariableOp0sequential_12_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential_12/h_2/MatMul/ReadVariableOp«
sequential_12/h_2/MatMulMatMul$sequential_12/h_1/Relu:activations:0/sequential_12/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/MatMul¬
(sequential_12/h_2/BiasAdd/ReadVariableOpReadVariableOp1sequential_12_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential_12/h_2/BiasAdd/ReadVariableOp…
sequential_12/h_2/BiasAddBiasAdd"sequential_12/h_2/MatMul:product:00sequential_12/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/BiasAddО
sequential_12/h_2/ReluRelu"sequential_12/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_12/h_2/Relu
IdentityIdentity$sequential_12/h_2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityш
NoOpNoOp)^sequential_12/h_1/BiasAdd/ReadVariableOp(^sequential_12/h_1/MatMul/ReadVariableOp)^sequential_12/h_2/BiasAdd/ReadVariableOp(^sequential_12/h_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2T
(sequential_12/h_1/BiasAdd/ReadVariableOp(sequential_12/h_1/BiasAdd/ReadVariableOp2R
'sequential_12/h_1/MatMul/ReadVariableOp'sequential_12/h_1/MatMul/ReadVariableOp2T
(sequential_12/h_2/BiasAdd/ReadVariableOp(sequential_12/h_2/BiasAdd/ReadVariableOp2R
'sequential_12/h_2/MatMul/ReadVariableOp'sequential_12/h_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
вN
Ѕ
!__inference__wrapped_model_501165
input_1T
Bmdn_freq_seqblock_sequential_12_h_1_matmul_readvariableop_resource:Q
Cmdn_freq_seqblock_sequential_12_h_1_biasadd_readvariableop_resource:T
Bmdn_freq_seqblock_sequential_12_h_2_matmul_readvariableop_resource:Q
Cmdn_freq_seqblock_sequential_12_h_2_biasadd_readvariableop_resource:F
4mdn_freq_output_layer_matmul_readvariableop_resource:C
5mdn_freq_output_layer_biasadd_readvariableop_resource:<
*mdn_freq_pi_matmul_readvariableop_resource:9
+mdn_freq_pi_biasadd_readvariableop_resource:<
*mdn_freq_mu_matmul_readvariableop_resource:9
+mdn_freq_mu_biasadd_readvariableop_resource:?
-mdn_freq_delta_matmul_readvariableop_resource:<
.mdn_freq_delta_biasadd_readvariableop_resource:
identityИҐ:MDN_freq/SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOpҐ9MDN_freq/SeqBlock/sequential_12/h_1/MatMul/ReadVariableOpҐ:MDN_freq/SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOpҐ9MDN_freq/SeqBlock/sequential_12/h_2/MatMul/ReadVariableOpҐ%MDN_freq/delta/BiasAdd/ReadVariableOpҐ$MDN_freq/delta/MatMul/ReadVariableOpҐ"MDN_freq/mu/BiasAdd/ReadVariableOpҐ!MDN_freq/mu/MatMul/ReadVariableOpҐ,MDN_freq/output_layer/BiasAdd/ReadVariableOpҐ+MDN_freq/output_layer/MatMul/ReadVariableOpҐ"MDN_freq/pi/BiasAdd/ReadVariableOpҐ!MDN_freq/pi/MatMul/ReadVariableOpщ
9MDN_freq/SeqBlock/sequential_12/h_1/MatMul/ReadVariableOpReadVariableOpBmdn_freq_seqblock_sequential_12_h_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02;
9MDN_freq/SeqBlock/sequential_12/h_1/MatMul/ReadVariableOpа
*MDN_freq/SeqBlock/sequential_12/h_1/MatMulMatMulinput_1AMDN_freq/SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2,
*MDN_freq/SeqBlock/sequential_12/h_1/MatMulш
:MDN_freq/SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOpReadVariableOpCmdn_freq_seqblock_sequential_12_h_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:MDN_freq/SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOpС
+MDN_freq/SeqBlock/sequential_12/h_1/BiasAddBiasAdd4MDN_freq/SeqBlock/sequential_12/h_1/MatMul:product:0BMDN_freq/SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2-
+MDN_freq/SeqBlock/sequential_12/h_1/BiasAddƒ
(MDN_freq/SeqBlock/sequential_12/h_1/ReluRelu4MDN_freq/SeqBlock/sequential_12/h_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2*
(MDN_freq/SeqBlock/sequential_12/h_1/Reluщ
9MDN_freq/SeqBlock/sequential_12/h_2/MatMul/ReadVariableOpReadVariableOpBmdn_freq_seqblock_sequential_12_h_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02;
9MDN_freq/SeqBlock/sequential_12/h_2/MatMul/ReadVariableOpП
*MDN_freq/SeqBlock/sequential_12/h_2/MatMulMatMul6MDN_freq/SeqBlock/sequential_12/h_1/Relu:activations:0AMDN_freq/SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2,
*MDN_freq/SeqBlock/sequential_12/h_2/MatMulш
:MDN_freq/SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOpReadVariableOpCmdn_freq_seqblock_sequential_12_h_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:MDN_freq/SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOpС
+MDN_freq/SeqBlock/sequential_12/h_2/BiasAddBiasAdd4MDN_freq/SeqBlock/sequential_12/h_2/MatMul:product:0BMDN_freq/SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2-
+MDN_freq/SeqBlock/sequential_12/h_2/BiasAddƒ
(MDN_freq/SeqBlock/sequential_12/h_2/ReluRelu4MDN_freq/SeqBlock/sequential_12/h_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2*
(MDN_freq/SeqBlock/sequential_12/h_2/Reluѕ
+MDN_freq/output_layer/MatMul/ReadVariableOpReadVariableOp4mdn_freq_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+MDN_freq/output_layer/MatMul/ReadVariableOpе
MDN_freq/output_layer/MatMulMatMul6MDN_freq/SeqBlock/sequential_12/h_2/Relu:activations:03MDN_freq/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/output_layer/MatMulќ
,MDN_freq/output_layer/BiasAdd/ReadVariableOpReadVariableOp5mdn_freq_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,MDN_freq/output_layer/BiasAdd/ReadVariableOpў
MDN_freq/output_layer/BiasAddBiasAdd&MDN_freq/output_layer/MatMul:product:04MDN_freq/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/output_layer/BiasAddЪ
MDN_freq/output_layer/ReluRelu&MDN_freq/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/output_layer/Relu±
!MDN_freq/pi/MatMul/ReadVariableOpReadVariableOp*mdn_freq_pi_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!MDN_freq/pi/MatMul/ReadVariableOpє
MDN_freq/pi/MatMulMatMul(MDN_freq/output_layer/Relu:activations:0)MDN_freq/pi/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/pi/MatMul∞
"MDN_freq/pi/BiasAdd/ReadVariableOpReadVariableOp+mdn_freq_pi_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"MDN_freq/pi/BiasAdd/ReadVariableOp±
MDN_freq/pi/BiasAddBiasAddMDN_freq/pi/MatMul:product:0*MDN_freq/pi/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/pi/BiasAddЕ
MDN_freq/pi/SigmoidSigmoidMDN_freq/pi/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/pi/Sigmoid±
!MDN_freq/mu/MatMul/ReadVariableOpReadVariableOp*mdn_freq_mu_matmul_readvariableop_resource*
_output_shapes

:*
dtype02#
!MDN_freq/mu/MatMul/ReadVariableOpє
MDN_freq/mu/MatMulMatMul(MDN_freq/output_layer/Relu:activations:0)MDN_freq/mu/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/mu/MatMul∞
"MDN_freq/mu/BiasAdd/ReadVariableOpReadVariableOp+mdn_freq_mu_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"MDN_freq/mu/BiasAdd/ReadVariableOp±
MDN_freq/mu/BiasAddBiasAddMDN_freq/mu/MatMul:product:0*MDN_freq/mu/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/mu/BiasAddИ
MDN_freq/mu/SoftplusSoftplusMDN_freq/mu/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/mu/SoftplusЇ
$MDN_freq/delta/MatMul/ReadVariableOpReadVariableOp-mdn_freq_delta_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$MDN_freq/delta/MatMul/ReadVariableOp¬
MDN_freq/delta/MatMulMatMul(MDN_freq/output_layer/Relu:activations:0,MDN_freq/delta/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/delta/MatMulє
%MDN_freq/delta/BiasAdd/ReadVariableOpReadVariableOp.mdn_freq_delta_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%MDN_freq/delta/BiasAdd/ReadVariableOpљ
MDN_freq/delta/BiasAddBiasAddMDN_freq/delta/MatMul:product:0-MDN_freq/delta/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/delta/BiasAddС
MDN_freq/delta/SoftplusSoftplusMDN_freq/delta/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/delta/Softplusx
MDN_freq/pvec/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
MDN_freq/pvec/concat/axisы
MDN_freq/pvec/concatConcatV2MDN_freq/pi/Sigmoid:y:0"MDN_freq/mu/Softplus:activations:0%MDN_freq/delta/Softplus:activations:0"MDN_freq/pvec/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€2
MDN_freq/pvec/concatx
IdentityIdentityMDN_freq/pvec/concat:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityю
NoOpNoOp;^MDN_freq/SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp:^MDN_freq/SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp;^MDN_freq/SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp:^MDN_freq/SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp&^MDN_freq/delta/BiasAdd/ReadVariableOp%^MDN_freq/delta/MatMul/ReadVariableOp#^MDN_freq/mu/BiasAdd/ReadVariableOp"^MDN_freq/mu/MatMul/ReadVariableOp-^MDN_freq/output_layer/BiasAdd/ReadVariableOp,^MDN_freq/output_layer/MatMul/ReadVariableOp#^MDN_freq/pi/BiasAdd/ReadVariableOp"^MDN_freq/pi/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 2x
:MDN_freq/SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp:MDN_freq/SeqBlock/sequential_12/h_1/BiasAdd/ReadVariableOp2v
9MDN_freq/SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp9MDN_freq/SeqBlock/sequential_12/h_1/MatMul/ReadVariableOp2x
:MDN_freq/SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp:MDN_freq/SeqBlock/sequential_12/h_2/BiasAdd/ReadVariableOp2v
9MDN_freq/SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp9MDN_freq/SeqBlock/sequential_12/h_2/MatMul/ReadVariableOp2N
%MDN_freq/delta/BiasAdd/ReadVariableOp%MDN_freq/delta/BiasAdd/ReadVariableOp2L
$MDN_freq/delta/MatMul/ReadVariableOp$MDN_freq/delta/MatMul/ReadVariableOp2H
"MDN_freq/mu/BiasAdd/ReadVariableOp"MDN_freq/mu/BiasAdd/ReadVariableOp2F
!MDN_freq/mu/MatMul/ReadVariableOp!MDN_freq/mu/MatMul/ReadVariableOp2\
,MDN_freq/output_layer/BiasAdd/ReadVariableOp,MDN_freq/output_layer/BiasAdd/ReadVariableOp2Z
+MDN_freq/output_layer/MatMul/ReadVariableOp+MDN_freq/output_layer/MatMul/ReadVariableOp2H
"MDN_freq/pi/BiasAdd/ReadVariableOp"MDN_freq/pi/BiasAdd/ReadVariableOp2F
!MDN_freq/pi/MatMul/ReadVariableOp!MDN_freq/pi/MatMul/ReadVariableOp:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
э

р
?__inference_h_1_layer_call_and_return_conditional_losses_502172

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
М
п
>__inference_mu_layer_call_and_return_conditional_losses_502055

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
нZ
Ж
__inference__traced_save_502350
file_prefix;
7savev2_mdn_freq_output_layer_kernel_read_readvariableop9
5savev2_mdn_freq_output_layer_bias_read_readvariableop1
-savev2_mdn_freq_pi_kernel_read_readvariableop/
+savev2_mdn_freq_pi_bias_read_readvariableop1
-savev2_mdn_freq_mu_kernel_read_readvariableop/
+savev2_mdn_freq_mu_bias_read_readvariableop4
0savev2_mdn_freq_delta_kernel_read_readvariableop2
.savev2_mdn_freq_delta_bias_read_readvariableop(
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
>savev2_adam_mdn_freq_output_layer_kernel_m_read_readvariableop@
<savev2_adam_mdn_freq_output_layer_bias_m_read_readvariableop8
4savev2_adam_mdn_freq_pi_kernel_m_read_readvariableop6
2savev2_adam_mdn_freq_pi_bias_m_read_readvariableop8
4savev2_adam_mdn_freq_mu_kernel_m_read_readvariableop6
2savev2_adam_mdn_freq_mu_bias_m_read_readvariableop;
7savev2_adam_mdn_freq_delta_kernel_m_read_readvariableop9
5savev2_adam_mdn_freq_delta_bias_m_read_readvariableop0
,savev2_adam_h_1_kernel_m_read_readvariableop.
*savev2_adam_h_1_bias_m_read_readvariableop0
,savev2_adam_h_2_kernel_m_read_readvariableop.
*savev2_adam_h_2_bias_m_read_readvariableopB
>savev2_adam_mdn_freq_output_layer_kernel_v_read_readvariableop@
<savev2_adam_mdn_freq_output_layer_bias_v_read_readvariableop8
4savev2_adam_mdn_freq_pi_kernel_v_read_readvariableop6
2savev2_adam_mdn_freq_pi_bias_v_read_readvariableop8
4savev2_adam_mdn_freq_mu_kernel_v_read_readvariableop6
2savev2_adam_mdn_freq_mu_bias_v_read_readvariableop;
7savev2_adam_mdn_freq_delta_kernel_v_read_readvariableop9
5savev2_adam_mdn_freq_delta_bias_v_read_readvariableop0
,savev2_adam_h_1_kernel_v_read_readvariableop.
*savev2_adam_h_1_bias_v_read_readvariableop0
,savev2_adam_h_2_kernel_v_read_readvariableop.
*savev2_adam_h_2_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameД
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ц
valueМBЙ.B*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB$pi/kernel/.ATTRIBUTES/VARIABLE_VALUEB"pi/bias/.ATTRIBUTES/VARIABLE_VALUEB$mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB"mu/bias/.ATTRIBUTES/VARIABLE_VALUEB'delta/kernel/.ATTRIBUTES/VARIABLE_VALUEB%delta/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@pi/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>pi/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@mu/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>mu/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdelta/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAdelta/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@pi/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>pi/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@mu/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>mu/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCdelta/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAdelta/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesд
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЌ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_mdn_freq_output_layer_kernel_read_readvariableop5savev2_mdn_freq_output_layer_bias_read_readvariableop-savev2_mdn_freq_pi_kernel_read_readvariableop+savev2_mdn_freq_pi_bias_read_readvariableop-savev2_mdn_freq_mu_kernel_read_readvariableop+savev2_mdn_freq_mu_bias_read_readvariableop0savev2_mdn_freq_delta_kernel_read_readvariableop.savev2_mdn_freq_delta_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop%savev2_h_1_kernel_read_readvariableop#savev2_h_1_bias_read_readvariableop%savev2_h_2_kernel_read_readvariableop#savev2_h_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop>savev2_adam_mdn_freq_output_layer_kernel_m_read_readvariableop<savev2_adam_mdn_freq_output_layer_bias_m_read_readvariableop4savev2_adam_mdn_freq_pi_kernel_m_read_readvariableop2savev2_adam_mdn_freq_pi_bias_m_read_readvariableop4savev2_adam_mdn_freq_mu_kernel_m_read_readvariableop2savev2_adam_mdn_freq_mu_bias_m_read_readvariableop7savev2_adam_mdn_freq_delta_kernel_m_read_readvariableop5savev2_adam_mdn_freq_delta_bias_m_read_readvariableop,savev2_adam_h_1_kernel_m_read_readvariableop*savev2_adam_h_1_bias_m_read_readvariableop,savev2_adam_h_2_kernel_m_read_readvariableop*savev2_adam_h_2_bias_m_read_readvariableop>savev2_adam_mdn_freq_output_layer_kernel_v_read_readvariableop<savev2_adam_mdn_freq_output_layer_bias_v_read_readvariableop4savev2_adam_mdn_freq_pi_kernel_v_read_readvariableop2savev2_adam_mdn_freq_pi_bias_v_read_readvariableop4savev2_adam_mdn_freq_mu_kernel_v_read_readvariableop2savev2_adam_mdn_freq_mu_bias_v_read_readvariableop7savev2_adam_mdn_freq_delta_kernel_v_read_readvariableop5savev2_adam_mdn_freq_delta_bias_v_read_readvariableop,savev2_adam_h_1_kernel_v_read_readvariableop*savev2_adam_h_1_bias_v_read_readvariableop,savev2_adam_h_2_kernel_v_read_readvariableop*savev2_adam_h_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*Ћ
_input_shapesє
ґ: ::::::::: : : : : ::::: : : : ::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	
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

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::
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

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::.

_output_shapes
: 
З
Ф
I__inference_sequential_12_layer_call_and_return_conditional_losses_501207

inputs

h_1_501184:

h_1_501186:

h_2_501201:

h_2_501203:
identityИҐh_1/StatefulPartitionedCallҐh_2/StatefulPartitionedCallю
h_1/StatefulPartitionedCallStatefulPartitionedCallinputs
h_1_501184
h_1_501186*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_h_1_layer_call_and_return_conditional_losses_5011832
h_1/StatefulPartitionedCallЬ
h_2/StatefulPartitionedCallStatefulPartitionedCall$h_1/StatefulPartitionedCall:output:0
h_2_501201
h_2_501203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_h_2_layer_call_and_return_conditional_losses_5012002
h_2/StatefulPartitionedCall
IdentityIdentity$h_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityК
NoOpNoOp^h_1/StatefulPartitionedCall^h_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 2:
h_1/StatefulPartitionedCallh_1/StatefulPartitionedCall2:
h_2/StatefulPartitionedCallh_2/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ю

п
>__inference_pi_layer_call_and_return_conditional_losses_502035

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
џ
‘
.__inference_sequential_12_layer_call_fn_501291
	h_1_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCall	h_1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_5012672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	h_1_input
Оњ
‘
"__inference__traced_restore_502495
file_prefix?
-assignvariableop_mdn_freq_output_layer_kernel:;
-assignvariableop_1_mdn_freq_output_layer_bias:7
%assignvariableop_2_mdn_freq_pi_kernel:1
#assignvariableop_3_mdn_freq_pi_bias:7
%assignvariableop_4_mdn_freq_mu_kernel:1
#assignvariableop_5_mdn_freq_mu_bias::
(assignvariableop_6_mdn_freq_delta_kernel:4
&assignvariableop_7_mdn_freq_delta_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: 0
assignvariableop_13_h_1_kernel:*
assignvariableop_14_h_1_bias:0
assignvariableop_15_h_2_kernel:*
assignvariableop_16_h_2_bias:#
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: I
7assignvariableop_21_adam_mdn_freq_output_layer_kernel_m:C
5assignvariableop_22_adam_mdn_freq_output_layer_bias_m:?
-assignvariableop_23_adam_mdn_freq_pi_kernel_m:9
+assignvariableop_24_adam_mdn_freq_pi_bias_m:?
-assignvariableop_25_adam_mdn_freq_mu_kernel_m:9
+assignvariableop_26_adam_mdn_freq_mu_bias_m:B
0assignvariableop_27_adam_mdn_freq_delta_kernel_m:<
.assignvariableop_28_adam_mdn_freq_delta_bias_m:7
%assignvariableop_29_adam_h_1_kernel_m:1
#assignvariableop_30_adam_h_1_bias_m:7
%assignvariableop_31_adam_h_2_kernel_m:1
#assignvariableop_32_adam_h_2_bias_m:I
7assignvariableop_33_adam_mdn_freq_output_layer_kernel_v:C
5assignvariableop_34_adam_mdn_freq_output_layer_bias_v:?
-assignvariableop_35_adam_mdn_freq_pi_kernel_v:9
+assignvariableop_36_adam_mdn_freq_pi_bias_v:?
-assignvariableop_37_adam_mdn_freq_mu_kernel_v:9
+assignvariableop_38_adam_mdn_freq_mu_bias_v:B
0assignvariableop_39_adam_mdn_freq_delta_kernel_v:<
.assignvariableop_40_adam_mdn_freq_delta_bias_v:7
%assignvariableop_41_adam_h_1_kernel_v:1
#assignvariableop_42_adam_h_1_bias_v:7
%assignvariableop_43_adam_h_2_kernel_v:1
#assignvariableop_44_adam_h_2_bias_v:
identity_46ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9К
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ц
valueМBЙ.B*outlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB(outlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB$pi/kernel/.ATTRIBUTES/VARIABLE_VALUEB"pi/bias/.ATTRIBUTES/VARIABLE_VALUEB$mu/kernel/.ATTRIBUTES/VARIABLE_VALUEB"mu/bias/.ATTRIBUTES/VARIABLE_VALUEB'delta/kernel/.ATTRIBUTES/VARIABLE_VALUEB%delta/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@pi/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>pi/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@mu/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>mu/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCdelta/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAdelta/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFoutlayer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDoutlayer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@pi/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>pi/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@mu/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>mu/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCdelta/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAdelta/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesк
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesФ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ќ
_output_shapesї
Є::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityђ
AssignVariableOpAssignVariableOp-assignvariableop_mdn_freq_output_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1≤
AssignVariableOp_1AssignVariableOp-assignvariableop_1_mdn_freq_output_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2™
AssignVariableOp_2AssignVariableOp%assignvariableop_2_mdn_freq_pi_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3®
AssignVariableOp_3AssignVariableOp#assignvariableop_3_mdn_freq_pi_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4™
AssignVariableOp_4AssignVariableOp%assignvariableop_4_mdn_freq_mu_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5®
AssignVariableOp_5AssignVariableOp#assignvariableop_5_mdn_freq_mu_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6≠
AssignVariableOp_6AssignVariableOp(assignvariableop_6_mdn_freq_delta_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ђ
AssignVariableOp_7AssignVariableOp&assignvariableop_7_mdn_freq_delta_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8°
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9£
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10І
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ѓ
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¶
AssignVariableOp_13AssignVariableOpassignvariableop_13_h_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14§
AssignVariableOp_14AssignVariableOpassignvariableop_14_h_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¶
AssignVariableOp_15AssignVariableOpassignvariableop_15_h_2_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16§
AssignVariableOp_16AssignVariableOpassignvariableop_16_h_2_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17°
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18°
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19£
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21њ
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_mdn_freq_output_layer_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22љ
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adam_mdn_freq_output_layer_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23µ
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_mdn_freq_pi_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24≥
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_mdn_freq_pi_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25µ
AssignVariableOp_25AssignVariableOp-assignvariableop_25_adam_mdn_freq_mu_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26≥
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_mdn_freq_mu_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Є
AssignVariableOp_27AssignVariableOp0assignvariableop_27_adam_mdn_freq_delta_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ґ
AssignVariableOp_28AssignVariableOp.assignvariableop_28_adam_mdn_freq_delta_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29≠
AssignVariableOp_29AssignVariableOp%assignvariableop_29_adam_h_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ђ
AssignVariableOp_30AssignVariableOp#assignvariableop_30_adam_h_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31≠
AssignVariableOp_31AssignVariableOp%assignvariableop_31_adam_h_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ђ
AssignVariableOp_32AssignVariableOp#assignvariableop_32_adam_h_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33њ
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_mdn_freq_output_layer_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34љ
AssignVariableOp_34AssignVariableOp5assignvariableop_34_adam_mdn_freq_output_layer_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35µ
AssignVariableOp_35AssignVariableOp-assignvariableop_35_adam_mdn_freq_pi_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36≥
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_mdn_freq_pi_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37µ
AssignVariableOp_37AssignVariableOp-assignvariableop_37_adam_mdn_freq_mu_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38≥
AssignVariableOp_38AssignVariableOp+assignvariableop_38_adam_mdn_freq_mu_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Є
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_mdn_freq_delta_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ґ
AssignVariableOp_40AssignVariableOp.assignvariableop_40_adam_mdn_freq_delta_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41≠
AssignVariableOp_41AssignVariableOp%assignvariableop_41_adam_h_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ђ
AssignVariableOp_42AssignVariableOp#assignvariableop_42_adam_h_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43≠
AssignVariableOp_43AssignVariableOp%assignvariableop_43_adam_h_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ђ
AssignVariableOp_44AssignVariableOp#assignvariableop_44_adam_h_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЉ
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45f
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_46§
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
х

°
$__inference_signature_wrapper_501779
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_5011652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:€€€€€€€€€: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
“
—
.__inference_sequential_12_layer_call_fn_502103

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_5012072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
П
т
A__inference_delta_layer_call_and_return_conditional_losses_502075

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Softplusq
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ђ
serving_defaultЧ
;
input_10
serving_default_input_1:0€€€€€€€€€<
output_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ЯЃ
•
seqblock
outlayer
pi
mu
	delta
pvec
	optimizer
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
Ъ_default_save_signature
Ы__call__
+Ь&call_and_return_all_conditional_losses"
_tf_keras_model
і
nnmodel
trainable_variables
regularization_losses
	variables
	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
Я__call__
+†&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
°__call__
+Ґ&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
£__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
•__call__
+¶&call_and_return_all_conditional_losses"
_tf_keras_layer
І
*trainable_variables
+regularization_losses
,	variables
-	keras_api
І__call__
+®&call_and_return_all_conditional_losses"
_tf_keras_layer
√
.iter

/beta_1

0beta_2
	1decay
2learning_ratemВmГmДmЕmЖmЗ$mИ%mЙ3mК4mЛ5mМ6mНvОvПvРvСvТvУ$vФ%vХ3vЦ4vЧ5vШ6vЩ"
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
 "
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
ќ
7layer_metrics
8non_trainable_variables
9metrics

:layers
trainable_variables
;layer_regularization_losses
	regularization_losses

	variables
Ы__call__
Ъ_default_save_signature
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
-
©serving_default"
signature_map
ъ
<layer_with_weights-0
<layer-0
=layer_with_weights-1
=layer-1
>trainable_variables
?regularization_losses
@	variables
A	keras_api
™__call__
+Ђ&call_and_return_all_conditional_losses"
_tf_keras_sequential
<
30
41
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
∞
Blayer_metrics
Cnon_trainable_variables
Dmetrics

Elayers
trainable_variables
Flayer_regularization_losses
regularization_losses
	variables
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
.:,2MDN_freq/output_layer/kernel
(:&2MDN_freq/output_layer/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
Glayer_metrics
Hnon_trainable_variables
Imetrics

Jlayers
trainable_variables
Klayer_regularization_losses
regularization_losses
	variables
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
$:"2MDN_freq/pi/kernel
:2MDN_freq/pi/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
Llayer_metrics
Mnon_trainable_variables
Nmetrics

Olayers
trainable_variables
Player_regularization_losses
regularization_losses
	variables
°__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
$:"2MDN_freq/mu/kernel
:2MDN_freq/mu/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
Qlayer_metrics
Rnon_trainable_variables
Smetrics

Tlayers
 trainable_variables
Ulayer_regularization_losses
!regularization_losses
"	variables
£__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
':%2MDN_freq/delta/kernel
!:2MDN_freq/delta/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
∞
Vlayer_metrics
Wnon_trainable_variables
Xmetrics

Ylayers
&trainable_variables
Zlayer_regularization_losses
'regularization_losses
(	variables
•__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
[layer_metrics
\non_trainable_variables
]metrics

^layers
*trainable_variables
_layer_regularization_losses
+regularization_losses
,	variables
І__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
:2
h_1/kernel
:2h_1/bias
:2
h_2/kernel
:2h_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
`0
a1"
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
љ

3kernel
4bias
btrainable_variables
cregularization_losses
d	variables
e	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

5kernel
6bias
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
<
30
41
52
63"
trackable_list_wrapper
 "
trackable_list_wrapper
<
30
41
52
63"
trackable_list_wrapper
∞
jlayer_metrics
knon_trainable_variables
lmetrics

mlayers
>trainable_variables
nlayer_regularization_losses
?regularization_losses
@	variables
™__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
N
	ototal
	pcount
q	variables
r	keras_api"
_tf_keras_metric
^
	stotal
	tcount
u
_fn_kwargs
v	variables
w	keras_api"
_tf_keras_metric
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
∞
xlayer_metrics
ynon_trainable_variables
zmetrics

{layers
btrainable_variables
|layer_regularization_losses
cregularization_losses
d	variables
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
≤
}layer_metrics
~non_trainable_variables
metrics
Аlayers
ftrainable_variables
 Бlayer_regularization_losses
gregularization_losses
h	variables
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
o0
p1"
trackable_list_wrapper
-
q	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
s0
t1"
trackable_list_wrapper
-
v	variables"
_generic_user_object
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
3:12#Adam/MDN_freq/output_layer/kernel/m
-:+2!Adam/MDN_freq/output_layer/bias/m
):'2Adam/MDN_freq/pi/kernel/m
#:!2Adam/MDN_freq/pi/bias/m
):'2Adam/MDN_freq/mu/kernel/m
#:!2Adam/MDN_freq/mu/bias/m
,:*2Adam/MDN_freq/delta/kernel/m
&:$2Adam/MDN_freq/delta/bias/m
!:2Adam/h_1/kernel/m
:2Adam/h_1/bias/m
!:2Adam/h_2/kernel/m
:2Adam/h_2/bias/m
3:12#Adam/MDN_freq/output_layer/kernel/v
-:+2!Adam/MDN_freq/output_layer/bias/v
):'2Adam/MDN_freq/pi/kernel/v
#:!2Adam/MDN_freq/pi/bias/v
):'2Adam/MDN_freq/mu/kernel/v
#:!2Adam/MDN_freq/mu/bias/v
,:*2Adam/MDN_freq/delta/kernel/v
&:$2Adam/MDN_freq/delta/bias/v
!:2Adam/h_1/kernel/v
:2Adam/h_1/bias/v
!:2Adam/h_2/kernel/v
:2Adam/h_2/bias/v
ћB…
!__inference__wrapped_model_501165input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
е2в
)__inference_MDN_freq_layer_call_fn_501460
)__inference_MDN_freq_layer_call_fn_501808
)__inference_MDN_freq_layer_call_fn_501837
)__inference_MDN_freq_layer_call_fn_501674≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501885
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501933
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501708
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501742≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
П2М
)__inference_SeqBlock_layer_call_fn_501946
)__inference_SeqBlock_layer_call_fn_501959≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≈2¬
D__inference_SeqBlock_layer_call_and_return_conditional_losses_501977
D__inference_SeqBlock_layer_call_and_return_conditional_losses_501995≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_output_layer_layer_call_fn_502004Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_output_layer_layer_call_and_return_conditional_losses_502015Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ќ2 
#__inference_pi_layer_call_fn_502024Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
и2е
>__inference_pi_layer_call_and_return_conditional_losses_502035Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ќ2 
#__inference_mu_layer_call_fn_502044Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
и2е
>__inference_mu_layer_call_and_return_conditional_losses_502055Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_delta_layer_call_fn_502064Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_delta_layer_call_and_return_conditional_losses_502075Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_pvec_layer_call_fn_502082Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_pvec_layer_call_and_return_conditional_losses_502090Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЋB»
$__inference_signature_wrapper_501779input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ж2Г
.__inference_sequential_12_layer_call_fn_501218
.__inference_sequential_12_layer_call_fn_502103
.__inference_sequential_12_layer_call_fn_502116
.__inference_sequential_12_layer_call_fn_501291ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
т2п
I__inference_sequential_12_layer_call_and_return_conditional_losses_502134
I__inference_sequential_12_layer_call_and_return_conditional_losses_502152
I__inference_sequential_12_layer_call_and_return_conditional_losses_501305
I__inference_sequential_12_layer_call_and_return_conditional_losses_501319ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ќ2Ћ
$__inference_h_1_layer_call_fn_502161Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
й2ж
?__inference_h_1_layer_call_and_return_conditional_losses_502172Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќ2Ћ
$__inference_h_2_layer_call_fn_502181Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
й2ж
?__inference_h_2_layer_call_and_return_conditional_losses_502192Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ≥
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501708k3456$%4Ґ1
*Ґ'
!К
input_1€€€€€€€€€
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ≥
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501742k3456$%4Ґ1
*Ґ'
!К
input_1€€€€€€€€€
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ ≤
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501885j3456$%3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ≤
D__inference_MDN_freq_layer_call_and_return_conditional_losses_501933j3456$%3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ Л
)__inference_MDN_freq_layer_call_fn_501460^3456$%4Ґ1
*Ґ'
!К
input_1€€€€€€€€€
p 
™ "К€€€€€€€€€Л
)__inference_MDN_freq_layer_call_fn_501674^3456$%4Ґ1
*Ґ'
!К
input_1€€€€€€€€€
p
™ "К€€€€€€€€€К
)__inference_MDN_freq_layer_call_fn_501808]3456$%3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "К€€€€€€€€€К
)__inference_MDN_freq_layer_call_fn_501837]3456$%3Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "К€€€€€€€€€™
D__inference_SeqBlock_layer_call_and_return_conditional_losses_501977b34563Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ™
D__inference_SeqBlock_layer_call_and_return_conditional_losses_501995b34563Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "%Ґ"
К
0€€€€€€€€€
Ъ В
)__inference_SeqBlock_layer_call_fn_501946U34563Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p 
™ "К€€€€€€€€€В
)__inference_SeqBlock_layer_call_fn_501959U34563Ґ0
)Ґ&
 К
inputs€€€€€€€€€
p
™ "К€€€€€€€€€Ъ
!__inference__wrapped_model_501165u3456$%0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "3™0
.
output_1"К
output_1€€€€€€€€€°
A__inference_delta_layer_call_and_return_conditional_losses_502075\$%/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ y
&__inference_delta_layer_call_fn_502064O$%/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Я
?__inference_h_1_layer_call_and_return_conditional_losses_502172\34/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ w
$__inference_h_1_layer_call_fn_502161O34/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Я
?__inference_h_2_layer_call_and_return_conditional_losses_502192\56/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ w
$__inference_h_2_layer_call_fn_502181O56/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Ю
>__inference_mu_layer_call_and_return_conditional_losses_502055\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ v
#__inference_mu_layer_call_fn_502044O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€®
H__inference_output_layer_layer_call_and_return_conditional_losses_502015\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ А
-__inference_output_layer_layer_call_fn_502004O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Ю
>__inference_pi_layer_call_and_return_conditional_losses_502035\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ v
#__inference_pi_layer_call_fn_502024O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€м
@__inference_pvec_layer_call_and_return_conditional_losses_502090І~Ґ{
tҐq
oЪl
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ ƒ
%__inference_pvec_layer_call_fn_502082Ъ~Ґ{
tҐq
oЪl
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
"К
inputs/2€€€€€€€€€
™ "К€€€€€€€€€ґ
I__inference_sequential_12_layer_call_and_return_conditional_losses_501305i3456:Ґ7
0Ґ-
#К 
	h_1_input€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ґ
I__inference_sequential_12_layer_call_and_return_conditional_losses_501319i3456:Ґ7
0Ґ-
#К 
	h_1_input€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ≥
I__inference_sequential_12_layer_call_and_return_conditional_losses_502134f34567Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ≥
I__inference_sequential_12_layer_call_and_return_conditional_losses_502152f34567Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ О
.__inference_sequential_12_layer_call_fn_501218\3456:Ґ7
0Ґ-
#К 
	h_1_input€€€€€€€€€
p 

 
™ "К€€€€€€€€€О
.__inference_sequential_12_layer_call_fn_501291\3456:Ґ7
0Ґ-
#К 
	h_1_input€€€€€€€€€
p

 
™ "К€€€€€€€€€Л
.__inference_sequential_12_layer_call_fn_502103Y34567Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€Л
.__inference_sequential_12_layer_call_fn_502116Y34567Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€©
$__inference_signature_wrapper_501779А3456$%;Ґ8
Ґ 
1™.
,
input_1!К
input_1€€€€€€€€€"3™0
.
output_1"К
output_1€€€€€€€€€