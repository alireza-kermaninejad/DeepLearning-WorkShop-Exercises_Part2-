??2
??
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 ?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-0-g3f878cff5b68??/
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
?
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
?
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
?
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
?
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
?
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameblock2_conv1/kernel
?
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@?*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:?*
dtype0
?
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock2_conv2/kernel
?
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv1/kernel
?
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:?*
dtype0
?
block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv2/kernel
?
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv3/kernel
?
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:?*
dtype0
?
block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv1/kernel
?
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:?*
dtype0
?
block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv2/kernel
?
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:?*
dtype0
?
block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv3/kernel
?
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:?*
dtype0
?
block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv1/kernel
?
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:?*
dtype0
?
block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv2/kernel
?
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:?*
dtype0
?
block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv3/kernel
?
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:?*
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
?
random_crop_2/cond/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*,
shared_namerandom_crop_2/cond/StateVar
?
/random_crop_2/cond/StateVar/Read/ReadVariableOpReadVariableOprandom_crop_2/cond/StateVar*
_output_shapes
:*
dtype0	
?
random_contrast_2/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*+
shared_namerandom_contrast_2/StateVar
?
.random_contrast_2/StateVar/Read/ReadVariableOpReadVariableOprandom_contrast_2/StateVar*
_output_shapes
:*
dtype0	
?
RMSprop/dense_3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_nameRMSprop/dense_3/kernel/rms
?
.RMSprop/dense_3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/kernel/rms*
_output_shapes
:	?*
dtype0
?
RMSprop/dense_3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_3/bias/rms
?
,RMSprop/dense_3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_3/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/block5_conv1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*0
shared_name!RMSprop/block5_conv1/kernel/rms
?
3RMSprop/block5_conv1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/block5_conv1/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/block5_conv1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nameRMSprop/block5_conv1/bias/rms
?
1RMSprop/block5_conv1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/block5_conv1/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/block5_conv2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*0
shared_name!RMSprop/block5_conv2/kernel/rms
?
3RMSprop/block5_conv2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/block5_conv2/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/block5_conv2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nameRMSprop/block5_conv2/bias/rms
?
1RMSprop/block5_conv2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/block5_conv2/bias/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/block5_conv3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*0
shared_name!RMSprop/block5_conv3/kernel/rms
?
3RMSprop/block5_conv3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/block5_conv3/kernel/rms*(
_output_shapes
:??*
dtype0
?
RMSprop/block5_conv3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nameRMSprop/block5_conv3/bias/rms
?
1RMSprop/block5_conv3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/block5_conv3/bias/rms*
_output_shapes	
:?*
dtype0
Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"????َ??)\??

NoOpNoOp
??
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??Bߝ Bם
?
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer-5
layer_with_weights-1
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 

	keras_api* 

	keras_api* 
?
layer-0
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
 layer_with_weights-2
 layer-4
!layer_with_weights-3
!layer-5
"layer-6
#layer_with_weights-4
#layer-7
$layer_with_weights-5
$layer-8
%layer_with_weights-6
%layer-9
&layer-10
'layer_with_weights-7
'layer-11
(layer_with_weights-8
(layer-12
)layer_with_weights-9
)layer-13
*layer-14
+layer_with_weights-10
+layer-15
,layer_with_weights-11
,layer-16
-layer_with_weights-12
-layer-17
.layer-18
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses*
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses* 
?

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
?
Citer
	Ddecay
Elearning_rate
Fmomentum
Grho
;rms?
<rms?
\rms?
]rms?
^rms?
_rms?
`rms?
arms?*
?
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19
\20
]21
^22
_23
`24
a25
;26
<27*
<
\0
]1
^2
_3
`4
a5
;6
<7*
* 
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

gserving_default* 
* 
* 
?
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses* 
?
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r_random_generator
s__call__
*t&call_and_return_all_conditional_losses*
?
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y_random_generator
z__call__
*{&call_and_return_all_conditional_losses*
* 
* 
* 
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
?

Hkernel
Ibias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

Jkernel
Kbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

Lkernel
Mbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

Nkernel
Obias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

Pkernel
Qbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

Rkernel
Sbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

Tkernel
Ubias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

Vkernel
Wbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

Xkernel
Ybias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

Zkernel
[bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?

\kernel
]bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

^kernel
_bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?

`kernel
abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19
\20
]21
^22
_23
`24
a25*
.
\0
]1
^2
_3
`4
a5*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1*

;0
<1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
?
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19*
5
0
1
2
3
4
5
6*

?0
?1*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
n	variables
otrainable_variables
pregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses* 

?
_generator*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

?
_generator*
* 
* 
* 

0
1
2*
* 
* 
* 

H0
I1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

J0
K1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

L0
M1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

N0
O1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

P0
Q1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

R0
S1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

T0
U1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

V0
W1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

X0
Y1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

Z0
[1*
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

\0
]1*

\0
]1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

^0
_1*

^0
_1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

`0
a1*

`0
a1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
?
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19*
?
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12
)13
*14
+15
,16
-17
.18*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?
_state_var*
* 
* 
* 
* 
* 

?
_state_var*

H0
I1*
* 
* 
* 
* 

J0
K1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

L0
M1*
* 
* 
* 
* 

N0
O1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

P0
Q1*
* 
* 
* 
* 

R0
S1*
* 
* 
* 
* 

T0
U1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

V0
W1*
* 
* 
* 
* 

X0
Y1*
* 
* 
* 
* 

Z0
[1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
??
VARIABLE_VALUErandom_crop_2/cond/StateVarRlayer-3/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUErandom_contrast_2/StateVarRlayer-3/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUERMSprop/dense_3/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUERMSprop/dense_3/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUERMSprop/block5_conv1/kernel/rmsEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUERMSprop/block5_conv1/bias/rmsEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUERMSprop/block5_conv2/kernel/rmsEvariables/22/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUERMSprop/block5_conv2/bias/rmsEvariables/23/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUERMSprop/block5_conv3/kernel/rmsEvariables/24/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUERMSprop/block5_conv3/bias/rmsEvariables/25/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_8Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_8Constrandom_crop_2/cond/StateVarblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasdense_3/kerneldense_3/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_15491
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/random_crop_2/cond/StateVar/Read/ReadVariableOp.random_contrast_2/StateVar/Read/ReadVariableOp.RMSprop/dense_3/kernel/rms/Read/ReadVariableOp,RMSprop/dense_3/bias/rms/Read/ReadVariableOp3RMSprop/block5_conv1/kernel/rms/Read/ReadVariableOp1RMSprop/block5_conv1/bias/rms/Read/ReadVariableOp3RMSprop/block5_conv2/kernel/rms/Read/ReadVariableOp1RMSprop/block5_conv2/bias/rms/Read/ReadVariableOp3RMSprop/block5_conv3/kernel/rms/Read/ReadVariableOp1RMSprop/block5_conv3/bias/rms/Read/ReadVariableOpConst_1*<
Tin5
321			*
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
GPU2*0J 8? *'
f"R 
__inference__traced_save_17516
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhoblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biastotalcounttotal_1count_1random_crop_2/cond/StateVarrandom_contrast_2/StateVarRMSprop/dense_3/kernel/rmsRMSprop/dense_3/bias/rmsRMSprop/block5_conv1/kernel/rmsRMSprop/block5_conv1/bias/rmsRMSprop/block5_conv2/kernel/rmsRMSprop/block5_conv2/bias/rmsRMSprop/block5_conv3/kernel/rmsRMSprop/block5_conv3/bias/rms*;
Tin4
220*
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
GPU2*0J 8? **
f%R#
!__inference__traced_restore_17667??,
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12956
center_crop_3_input!
random_crop_2_12951:	
identity??%center_crop_3/StatefulPartitionedCall?%random_crop_2/StatefulPartitionedCall?
%center_crop_3/StatefulPartitionedCallStatefulPartitionedCallcenter_crop_3_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_center_crop_3_layer_call_and_return_conditional_losses_12486?
%random_crop_2/StatefulPartitionedCallStatefulPartitionedCall.center_crop_3/StatefulPartitionedCall:output:0random_crop_2_12951*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_random_crop_2_layer_call_and_return_conditional_losses_12654?
!random_contrast_2/PartitionedCallPartitionedCall.random_crop_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_12662?
IdentityIdentity*random_contrast_2/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp&^center_crop_3/StatefulPartitionedCall&^random_crop_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2N
%center_crop_3/StatefulPartitionedCall%center_crop_3/StatefulPartitionedCall2N
%random_crop_2/StatefulPartitionedCall%random_crop_2/StatefulPartitionedCall:f b
1
_output_shapes
:???????????
-
_user_specified_namecenter_crop_3_input
ޓ
?
*sequential_2_random_crop_2_cond_true_14737V
Hsequential_2_random_crop_2_cond_stateful_uniform_rngreadandskip_resource:	F
Bsequential_2_random_crop_2_cond_add_sequential_2_random_crop_2_subJ
Fsequential_2_random_crop_2_cond_add_1_sequential_2_random_crop_2_sub_1g
csequential_2_random_crop_2_cond_crop_to_bounding_box_shape_sequential_2_center_crop_3_cond_identity,
(sequential_2_random_crop_2_cond_identity??Bsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert?Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert?Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert?Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert??sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip?
6sequential_2/random_crop_2/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
4sequential_2/random_crop_2/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : z
4sequential_2/random_crop_2/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
6sequential_2/random_crop_2/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
5sequential_2/random_crop_2/cond/stateful_uniform/ProdProd?sequential_2/random_crop_2/cond/stateful_uniform/shape:output:0?sequential_2/random_crop_2/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: y
7sequential_2/random_crop_2/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
7sequential_2/random_crop_2/cond/stateful_uniform/Cast_1Cast>sequential_2/random_crop_2/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
?sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkipRngReadAndSkipHsequential_2_random_crop_2_cond_stateful_uniform_rngreadandskip_resource@sequential_2/random_crop_2/cond/stateful_uniform/Cast/x:output:0;sequential_2/random_crop_2/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:?
Dsequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential_2/random_crop_2/cond/stateful_uniform/strided_sliceStridedSliceGsequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip:value:0Msequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack:output:0Osequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_1:output:0Osequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
8sequential_2/random_crop_2/cond/stateful_uniform/BitcastBitcastGsequential_2/random_crop_2/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
Fsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1StridedSliceGsequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip:value:0Osequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack:output:0Qsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_1:output:0Qsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
:sequential_2/random_crop_2/cond/stateful_uniform/Bitcast_1BitcastIsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0v
4sequential_2/random_crop_2/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
0sequential_2/random_crop_2/cond/stateful_uniformStatelessRandomUniformIntV2?sequential_2/random_crop_2/cond/stateful_uniform/shape:output:0Csequential_2/random_crop_2/cond/stateful_uniform/Bitcast_1:output:0Asequential_2/random_crop_2/cond/stateful_uniform/Bitcast:output:0=sequential_2/random_crop_2/cond/stateful_uniform/alg:output:0=sequential_2/random_crop_2/cond/stateful_uniform/min:output:0=sequential_2/random_crop_2/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0}
3sequential_2/random_crop_2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_2/random_crop_2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/random_crop_2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential_2/random_crop_2/cond/strided_sliceStridedSlice9sequential_2/random_crop_2/cond/stateful_uniform:output:0<sequential_2/random_crop_2/cond/strided_slice/stack:output:0>sequential_2/random_crop_2/cond/strided_slice/stack_1:output:0>sequential_2/random_crop_2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_2/random_crop_2/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential_2/random_crop_2/cond/addAddV2Bsequential_2_random_crop_2_cond_add_sequential_2_random_crop_2_sub.sequential_2/random_crop_2/cond/add/y:output:0*
T0*
_output_shapes
: ?
#sequential_2/random_crop_2/cond/modFloorMod6sequential_2/random_crop_2/cond/strided_slice:output:0'sequential_2/random_crop_2/cond/add:z:0*
T0*
_output_shapes
: 
5sequential_2/random_crop_2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7sequential_2/random_crop_2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_2/random_crop_2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_2/random_crop_2/cond/strided_slice_1StridedSlice9sequential_2/random_crop_2/cond/stateful_uniform:output:0>sequential_2/random_crop_2/cond/strided_slice_1/stack:output:0@sequential_2/random_crop_2/cond/strided_slice_1/stack_1:output:0@sequential_2/random_crop_2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'sequential_2/random_crop_2/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
%sequential_2/random_crop_2/cond/add_1AddV2Fsequential_2_random_crop_2_cond_add_1_sequential_2_random_crop_2_sub_10sequential_2/random_crop_2/cond/add_1/y:output:0*
T0*
_output_shapes
: ?
%sequential_2/random_crop_2/cond/mod_1FloorMod8sequential_2/random_crop_2/cond/strided_slice_1:output:0)sequential_2/random_crop_2/cond/add_1:z:0*
T0*
_output_shapes
: ?
:sequential_2/random_crop_2/cond/crop_to_bounding_box/ShapeShapecsequential_2_random_crop_2_cond_crop_to_bounding_box_shape_sequential_2_center_crop_3_cond_identity*
T0*
_output_shapes
:?
<sequential_2/random_crop_2/cond/crop_to_bounding_box/unstackUnpackCsequential_2/random_crop_2/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num?
Csequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Asequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqualGreaterEqual)sequential_2/random_crop_2/cond/mod_1:z:0Lsequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Asequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Isequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Bsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertAssertEsequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual:z:0Rsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
Esequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Csequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual'sequential_2/random_crop_2/cond/mod:z:0Nsequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Ksequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertAssertGsequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1:z:0Tsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0C^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 }
:sequential_2/random_crop_2/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
8sequential_2/random_crop_2/cond/crop_to_bounding_box/addAddV2Csequential_2/random_crop_2/cond/crop_to_bounding_box/add/x:output:0)sequential_2/random_crop_2/cond/mod_1:z:0*
T0*
_output_shapes
: ?
@sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
>sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual	LessEqual<sequential_2/random_crop_2/cond/crop_to_bounding_box/add:z:0Isequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Ksequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertAssertBsequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual:z:0Tsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0E^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 
<sequential_2/random_crop_2/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
:sequential_2/random_crop_2/cond/crop_to_bounding_box/add_1AddV2Esequential_2/random_crop_2/cond/crop_to_bounding_box/add_1/x:output:0'sequential_2/random_crop_2/cond/mod:z:0*
T0*
_output_shapes
: ?
Bsequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
@sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1	LessEqual>sequential_2/random_crop_2/cond/crop_to_bounding_box/add_1:z:0Ksequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Ksequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/AssertAssertDsequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1:z:0Tsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0E^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
Gsequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependencyIdentitycsequential_2_random_crop_2_cond_crop_to_bounding_box_shape_sequential_2_center_crop_3_cond_identityC^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert*
T0*;
_class1
/-loc:@sequential_2/center_crop_3/cond/Identity*1
_output_shapes
:???????????~
<sequential_2/random_crop_2/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : ~
<sequential_2/random_crop_2/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
:sequential_2/random_crop_2/cond/crop_to_bounding_box/stackPackEsequential_2/random_crop_2/cond/crop_to_bounding_box/stack/0:output:0'sequential_2/random_crop_2/cond/mod:z:0)sequential_2/random_crop_2/cond/mod_1:z:0Esequential_2/random_crop_2/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
<sequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_1ShapePsequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Hsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Jsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Jsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_sliceStridedSliceEsequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_1:output:0Qsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack:output:0Ssequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Ssequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<sequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_2ShapePsequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Jsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Lsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Lsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1StridedSliceEsequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_2:output:0Ssequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Usequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Usequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :??
>sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
<sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1PackKsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice:output:0Gsequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/1:output:0Gsequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/2:output:0Msequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
:sequential_2/random_crop_2/cond/crop_to_bounding_box/SliceSlicePsequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependency:output:0Csequential_2/random_crop_2/cond/crop_to_bounding_box/stack:output:0Esequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
(sequential_2/random_crop_2/cond/IdentityIdentityCsequential_2/random_crop_2/cond/crop_to_bounding_box/Slice:output:0%^sequential_2/random_crop_2/cond/NoOp*
T0*1
_output_shapes
:????????????
$sequential_2/random_crop_2/cond/NoOpNoOpC^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert@^sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_2_random_crop_2_cond_identity1sequential_2/random_crop_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????2?
Bsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertBsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert2?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertDsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert2?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertDsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert2?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/AssertDsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert2?
?sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip?sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
,__inference_block2_conv1_layer_call_fn_17090

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_13080x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????pp?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????pp@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?'
?
cond_false_16838
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_shape_inputs
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM

cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: O
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :?f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Cf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?V
?
@__inference_vgg16_layer_call_and_return_conditional_losses_13839
input_7,
block1_conv1_13768:@ 
block1_conv1_13770:@,
block1_conv2_13773:@@ 
block1_conv2_13775:@-
block2_conv1_13779:@?!
block2_conv1_13781:	?.
block2_conv2_13784:??!
block2_conv2_13786:	?.
block3_conv1_13790:??!
block3_conv1_13792:	?.
block3_conv2_13795:??!
block3_conv2_13797:	?.
block3_conv3_13800:??!
block3_conv3_13802:	?.
block4_conv1_13806:??!
block4_conv1_13808:	?.
block4_conv2_13811:??!
block4_conv2_13813:	?.
block4_conv3_13816:??!
block4_conv3_13818:	?.
block5_conv1_13822:??!
block5_conv1_13824:	?.
block5_conv2_13827:??!
block5_conv2_13829:	?.
block5_conv3_13832:??!
block5_conv3_13834:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_7block1_conv1_13768block1_conv1_13770*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_13045?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_13773block1_conv2_13775*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_13062?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_12976?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_13779block2_conv1_13781*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_13080?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_13784block2_conv2_13786*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_13097?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_12988?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_13790block3_conv1_13792*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_13115?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_13795block3_conv2_13797*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_13132?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_13800block3_conv3_13802*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_13149?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_13000?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_13806block4_conv1_13808*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_13167?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_13811block4_conv2_13813*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_13184?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_13816block4_conv3_13818*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_13201?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_13012?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_13822block5_conv1_13824*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_13219?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_13827block5_conv2_13829*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_13236?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_13832block5_conv3_13834*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_13253?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_13024|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_7
?
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_12976

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?&
?	
B__inference_model_3_layer_call_and_return_conditional_losses_13939

inputs!
tf_nn_bias_add_3_biasadd_bias 
sequential_2_13865:	%
vgg16_13868:@
vgg16_13870:@%
vgg16_13872:@@
vgg16_13874:@&
vgg16_13876:@?
vgg16_13878:	?'
vgg16_13880:??
vgg16_13882:	?'
vgg16_13884:??
vgg16_13886:	?'
vgg16_13888:??
vgg16_13890:	?'
vgg16_13892:??
vgg16_13894:	?'
vgg16_13896:??
vgg16_13898:	?'
vgg16_13900:??
vgg16_13902:	?'
vgg16_13904:??
vgg16_13906:	?'
vgg16_13908:??
vgg16_13910:	?'
vgg16_13912:??
vgg16_13914:	?'
vgg16_13916:??
vgg16_13918:	? 
dense_3_13933:	?
dense_3_13935:
identity??dense_3/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?vgg16/StatefulPartitionedCall
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ?????
(tf.__operators__.getitem_3/strided_sliceStridedSliceinputs7tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask?
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:????????????
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall!tf.nn.bias_add_3/BiasAdd:output:0sequential_2_13865*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12665?
vgg16/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0vgg16_13868vgg16_13870vgg16_13872vgg16_13874vgg16_13876vgg16_13878vgg16_13880vgg16_13882vgg16_13884vgg16_13886vgg16_13888vgg16_13890vgg16_13892vgg16_13894vgg16_13896vgg16_13898vgg16_13900vgg16_13902vgg16_13904vgg16_13906vgg16_13908vgg16_13910vgg16_13912vgg16_13914vgg16_13916vgg16_13918*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_13261?
*global_average_pooling2d_3/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_13849?
dense_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0dense_3_13933dense_3_13935*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_13932w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_3/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_13012

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?'
?
cond_false_16672
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_shape_inputs
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM

cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: O
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :?f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Cf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?V
?
@__inference_vgg16_layer_call_and_return_conditional_losses_13261

inputs,
block1_conv1_13046:@ 
block1_conv1_13048:@,
block1_conv2_13063:@@ 
block1_conv2_13065:@-
block2_conv1_13081:@?!
block2_conv1_13083:	?.
block2_conv2_13098:??!
block2_conv2_13100:	?.
block3_conv1_13116:??!
block3_conv1_13118:	?.
block3_conv2_13133:??!
block3_conv2_13135:	?.
block3_conv3_13150:??!
block3_conv3_13152:	?.
block4_conv1_13168:??!
block4_conv1_13170:	?.
block4_conv2_13185:??!
block4_conv2_13187:	?.
block4_conv3_13202:??!
block4_conv3_13204:	?.
block5_conv1_13220:??!
block5_conv1_13222:	?.
block5_conv2_13237:??!
block5_conv2_13239:	?.
block5_conv3_13254:??!
block5_conv3_13256:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_13046block1_conv1_13048*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_13045?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_13063block1_conv2_13065*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_13062?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_12976?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_13081block2_conv1_13083*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_13080?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_13098block2_conv2_13100*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_13097?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_12988?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_13116block3_conv1_13118*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_13115?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_13133block3_conv2_13135*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_13132?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_13150block3_conv3_13152*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_13149?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_13000?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_13168block4_conv1_13170*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_13167?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_13185block4_conv2_13187*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_13184?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_13202block4_conv3_13204*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_13201?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_13012?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_13220block5_conv1_13222*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_13219?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_13237block5_conv2_13239*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_13236?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_13254block5_conv3_13256*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_13253?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_13024|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?'
?
cond_false_12764
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_shape_inputs
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM

cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: O
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :?f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Cf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
%__inference_vgg16_layer_call_fn_13316
input_7!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_13261x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_7
?
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_17081

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?3
?
random_crop_2_cond_false_15670"
random_crop_2_cond_placeholder$
 random_crop_2_cond_placeholder_1$
 random_crop_2_cond_placeholder_28
4random_crop_2_cond_shape_center_crop_3_cond_identity
random_crop_2_cond_identity|
random_crop_2/cond/ShapeShape4random_crop_2_cond_shape_center_crop_3_cond_identity*
T0*
_output_shapes
:y
&random_crop_2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
(random_crop_2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(random_crop_2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop_2/cond/strided_sliceStridedSlice!random_crop_2/cond/Shape:output:0/random_crop_2/cond/strided_slice/stack:output:01random_crop_2/cond/strided_slice/stack_1:output:01random_crop_2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
(random_crop_2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*random_crop_2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*random_crop_2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"random_crop_2/cond/strided_slice_1StridedSlice!random_crop_2/cond/Shape:output:01random_crop_2/cond/strided_slice_1/stack:output:03random_crop_2/cond/strided_slice_1/stack_1:output:03random_crop_2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
random_crop_2/cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
random_crop_2/cond/mulMul+random_crop_2/cond/strided_slice_1:output:0!random_crop_2/cond/mul/y:output:0*
T0*
_output_shapes
: k
random_crop_2/cond/CastCastrandom_crop_2/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop_2/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
random_crop_2/cond/truedivRealDivrandom_crop_2/cond/Cast:y:0%random_crop_2/cond/truediv/y:output:0*
T0*
_output_shapes
: q
random_crop_2/cond/Cast_1Castrandom_crop_2/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: ]
random_crop_2/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
random_crop_2/cond/mul_1Mul)random_crop_2/cond/strided_slice:output:0#random_crop_2/cond/mul_1/y:output:0*
T0*
_output_shapes
: o
random_crop_2/cond/Cast_2Castrandom_crop_2/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_2/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
random_crop_2/cond/truediv_1RealDivrandom_crop_2/cond/Cast_2:y:0'random_crop_2/cond/truediv_1/y:output:0*
T0*
_output_shapes
: s
random_crop_2/cond/Cast_3Cast random_crop_2/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
random_crop_2/cond/MinimumMinimum)random_crop_2/cond/strided_slice:output:0random_crop_2/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
random_crop_2/cond/Minimum_1Minimum+random_crop_2/cond/strided_slice_1:output:0random_crop_2/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
random_crop_2/cond/subSub)random_crop_2/cond/strided_slice:output:0random_crop_2/cond/Minimum:z:0*
T0*
_output_shapes
: m
random_crop_2/cond/Cast_4Castrandom_crop_2/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_2/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
random_crop_2/cond/truediv_2RealDivrandom_crop_2/cond/Cast_4:y:0'random_crop_2/cond/truediv_2/y:output:0*
T0*
_output_shapes
: s
random_crop_2/cond/Cast_5Cast random_crop_2/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
random_crop_2/cond/sub_1Sub+random_crop_2/cond/strided_slice_1:output:0 random_crop_2/cond/Minimum_1:z:0*
T0*
_output_shapes
: o
random_crop_2/cond/Cast_6Castrandom_crop_2/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_2/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
random_crop_2/cond/truediv_3RealDivrandom_crop_2/cond/Cast_6:y:0'random_crop_2/cond/truediv_3/y:output:0*
T0*
_output_shapes
: s
random_crop_2/cond/Cast_7Cast random_crop_2/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: \
random_crop_2/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : \
random_crop_2/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
random_crop_2/cond/stackPack#random_crop_2/cond/stack/0:output:0random_crop_2/cond/Cast_5:y:0random_crop_2/cond/Cast_7:y:0#random_crop_2/cond/stack/3:output:0*
N*
T0*
_output_shapes
:g
random_crop_2/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????g
random_crop_2/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
random_crop_2/cond/stack_1Pack%random_crop_2/cond/stack_1/0:output:0random_crop_2/cond/Minimum:z:0 random_crop_2/cond/Minimum_1:z:0%random_crop_2/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
random_crop_2/cond/SliceSlice4random_crop_2_cond_shape_center_crop_3_cond_identity!random_crop_2/cond/stack:output:0#random_crop_2/cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????o
random_crop_2/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
(random_crop_2/cond/resize/ResizeBilinearResizeBilinear!random_crop_2/cond/Slice:output:0'random_crop_2/cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
random_crop_2/cond/IdentityIdentity9random_crop_2/cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"C
random_crop_2_cond_identity$random_crop_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
G__inference_block5_conv3_layer_call_and_return_conditional_losses_17331

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
-__inference_random_crop_2_layer_call_fn_16642

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_random_crop_2_layer_call_and_return_conditional_losses_12900y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block3_conv2_layer_call_and_return_conditional_losses_17171

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????88?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_17201

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block4_conv3_layer_call_and_return_conditional_losses_17261

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?C
?
3model_3_sequential_2_center_crop_3_cond_false_119677
3model_3_sequential_2_center_crop_3_cond_placeholder9
5model_3_sequential_2_center_crop_3_cond_placeholder_1R
Nmodel_3_sequential_2_center_crop_3_cond_shape_model_3_tf_nn_bias_add_3_biasadd4
0model_3_sequential_2_center_crop_3_cond_identity?
-model_3/sequential_2/center_crop_3/cond/ShapeShapeNmodel_3_sequential_2_center_crop_3_cond_shape_model_3_tf_nn_bias_add_3_biasadd*
T0*
_output_shapes
:?
;model_3/sequential_2/center_crop_3/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
=model_3/sequential_2/center_crop_3/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
=model_3/sequential_2/center_crop_3/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5model_3/sequential_2/center_crop_3/cond/strided_sliceStridedSlice6model_3/sequential_2/center_crop_3/cond/Shape:output:0Dmodel_3/sequential_2/center_crop_3/cond/strided_slice/stack:output:0Fmodel_3/sequential_2/center_crop_3/cond/strided_slice/stack_1:output:0Fmodel_3/sequential_2/center_crop_3/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
=model_3/sequential_2/center_crop_3/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
?model_3/sequential_2/center_crop_3/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
?model_3/sequential_2/center_crop_3/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7model_3/sequential_2/center_crop_3/cond/strided_slice_1StridedSlice6model_3/sequential_2/center_crop_3/cond/Shape:output:0Fmodel_3/sequential_2/center_crop_3/cond/strided_slice_1/stack:output:0Hmodel_3/sequential_2/center_crop_3/cond/strided_slice_1/stack_1:output:0Hmodel_3/sequential_2/center_crop_3/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
-model_3/sequential_2/center_crop_3/cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
+model_3/sequential_2/center_crop_3/cond/mulMul@model_3/sequential_2/center_crop_3/cond/strided_slice_1:output:06model_3/sequential_2/center_crop_3/cond/mul/y:output:0*
T0*
_output_shapes
: ?
,model_3/sequential_2/center_crop_3/cond/CastCast/model_3/sequential_2/center_crop_3/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: v
1model_3/sequential_2/center_crop_3/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
/model_3/sequential_2/center_crop_3/cond/truedivRealDiv0model_3/sequential_2/center_crop_3/cond/Cast:y:0:model_3/sequential_2/center_crop_3/cond/truediv/y:output:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/center_crop_3/cond/Cast_1Cast3model_3/sequential_2/center_crop_3/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: r
/model_3/sequential_2/center_crop_3/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
-model_3/sequential_2/center_crop_3/cond/mul_1Mul>model_3/sequential_2/center_crop_3/cond/strided_slice:output:08model_3/sequential_2/center_crop_3/cond/mul_1/y:output:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/center_crop_3/cond/Cast_2Cast1model_3/sequential_2/center_crop_3/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: x
3model_3/sequential_2/center_crop_3/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
1model_3/sequential_2/center_crop_3/cond/truediv_1RealDiv2model_3/sequential_2/center_crop_3/cond/Cast_2:y:0<model_3/sequential_2/center_crop_3/cond/truediv_1/y:output:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/center_crop_3/cond/Cast_3Cast5model_3/sequential_2/center_crop_3/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
/model_3/sequential_2/center_crop_3/cond/MinimumMinimum>model_3/sequential_2/center_crop_3/cond/strided_slice:output:02model_3/sequential_2/center_crop_3/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
1model_3/sequential_2/center_crop_3/cond/Minimum_1Minimum@model_3/sequential_2/center_crop_3/cond/strided_slice_1:output:02model_3/sequential_2/center_crop_3/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
+model_3/sequential_2/center_crop_3/cond/subSub>model_3/sequential_2/center_crop_3/cond/strided_slice:output:03model_3/sequential_2/center_crop_3/cond/Minimum:z:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/center_crop_3/cond/Cast_4Cast/model_3/sequential_2/center_crop_3/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: x
3model_3/sequential_2/center_crop_3/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
1model_3/sequential_2/center_crop_3/cond/truediv_2RealDiv2model_3/sequential_2/center_crop_3/cond/Cast_4:y:0<model_3/sequential_2/center_crop_3/cond/truediv_2/y:output:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/center_crop_3/cond/Cast_5Cast5model_3/sequential_2/center_crop_3/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
-model_3/sequential_2/center_crop_3/cond/sub_1Sub@model_3/sequential_2/center_crop_3/cond/strided_slice_1:output:05model_3/sequential_2/center_crop_3/cond/Minimum_1:z:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/center_crop_3/cond/Cast_6Cast1model_3/sequential_2/center_crop_3/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: x
3model_3/sequential_2/center_crop_3/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
1model_3/sequential_2/center_crop_3/cond/truediv_3RealDiv2model_3/sequential_2/center_crop_3/cond/Cast_6:y:0<model_3/sequential_2/center_crop_3/cond/truediv_3/y:output:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/center_crop_3/cond/Cast_7Cast5model_3/sequential_2/center_crop_3/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: q
/model_3/sequential_2/center_crop_3/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : q
/model_3/sequential_2/center_crop_3/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
-model_3/sequential_2/center_crop_3/cond/stackPack8model_3/sequential_2/center_crop_3/cond/stack/0:output:02model_3/sequential_2/center_crop_3/cond/Cast_5:y:02model_3/sequential_2/center_crop_3/cond/Cast_7:y:08model_3/sequential_2/center_crop_3/cond/stack/3:output:0*
N*
T0*
_output_shapes
:|
1model_3/sequential_2/center_crop_3/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????|
1model_3/sequential_2/center_crop_3/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
/model_3/sequential_2/center_crop_3/cond/stack_1Pack:model_3/sequential_2/center_crop_3/cond/stack_1/0:output:03model_3/sequential_2/center_crop_3/cond/Minimum:z:05model_3/sequential_2/center_crop_3/cond/Minimum_1:z:0:model_3/sequential_2/center_crop_3/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
-model_3/sequential_2/center_crop_3/cond/SliceSliceNmodel_3_sequential_2_center_crop_3_cond_shape_model_3_tf_nn_bias_add_3_biasadd6model_3/sequential_2/center_crop_3/cond/stack:output:08model_3/sequential_2/center_crop_3/cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$?????????????????????
3model_3/sequential_2/center_crop_3/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
=model_3/sequential_2/center_crop_3/cond/resize/ResizeBilinearResizeBilinear6model_3/sequential_2/center_crop_3/cond/Slice:output:0<model_3/sequential_2/center_crop_3/cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
0model_3/sequential_2/center_crop_3/cond/IdentityIdentityNmodel_3/sequential_2/center_crop_3/cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"m
0model_3_sequential_2_center_crop_3_cond_identity9model_3/sequential_2/center_crop_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
G__inference_block3_conv3_layer_call_and_return_conditional_losses_13149

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????88?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?=
?
+sequential_2_random_crop_2_cond_false_14738/
+sequential_2_random_crop_2_cond_placeholder1
-sequential_2_random_crop_2_cond_placeholder_11
-sequential_2_random_crop_2_cond_placeholder_2R
Nsequential_2_random_crop_2_cond_shape_sequential_2_center_crop_3_cond_identity,
(sequential_2_random_crop_2_cond_identity?
%sequential_2/random_crop_2/cond/ShapeShapeNsequential_2_random_crop_2_cond_shape_sequential_2_center_crop_3_cond_identity*
T0*
_output_shapes
:?
3sequential_2/random_crop_2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
5sequential_2/random_crop_2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????
5sequential_2/random_crop_2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential_2/random_crop_2/cond/strided_sliceStridedSlice.sequential_2/random_crop_2/cond/Shape:output:0<sequential_2/random_crop_2/cond/strided_slice/stack:output:0>sequential_2/random_crop_2/cond/strided_slice/stack_1:output:0>sequential_2/random_crop_2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_2/random_crop_2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
7sequential_2/random_crop_2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
7sequential_2/random_crop_2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_2/random_crop_2/cond/strided_slice_1StridedSlice.sequential_2/random_crop_2/cond/Shape:output:0>sequential_2/random_crop_2/cond/strided_slice_1/stack:output:0@sequential_2/random_crop_2/cond/strided_slice_1/stack_1:output:0@sequential_2/random_crop_2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%sequential_2/random_crop_2/cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
#sequential_2/random_crop_2/cond/mulMul8sequential_2/random_crop_2/cond/strided_slice_1:output:0.sequential_2/random_crop_2/cond/mul/y:output:0*
T0*
_output_shapes
: ?
$sequential_2/random_crop_2/cond/CastCast'sequential_2/random_crop_2/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: n
)sequential_2/random_crop_2/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
'sequential_2/random_crop_2/cond/truedivRealDiv(sequential_2/random_crop_2/cond/Cast:y:02sequential_2/random_crop_2/cond/truediv/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_1Cast+sequential_2/random_crop_2/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: j
'sequential_2/random_crop_2/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
%sequential_2/random_crop_2/cond/mul_1Mul6sequential_2/random_crop_2/cond/strided_slice:output:00sequential_2/random_crop_2/cond/mul_1/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_2Cast)sequential_2/random_crop_2/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/random_crop_2/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
)sequential_2/random_crop_2/cond/truediv_1RealDiv*sequential_2/random_crop_2/cond/Cast_2:y:04sequential_2/random_crop_2/cond/truediv_1/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_3Cast-sequential_2/random_crop_2/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
'sequential_2/random_crop_2/cond/MinimumMinimum6sequential_2/random_crop_2/cond/strided_slice:output:0*sequential_2/random_crop_2/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
)sequential_2/random_crop_2/cond/Minimum_1Minimum8sequential_2/random_crop_2/cond/strided_slice_1:output:0*sequential_2/random_crop_2/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
#sequential_2/random_crop_2/cond/subSub6sequential_2/random_crop_2/cond/strided_slice:output:0+sequential_2/random_crop_2/cond/Minimum:z:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_4Cast'sequential_2/random_crop_2/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/random_crop_2/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
)sequential_2/random_crop_2/cond/truediv_2RealDiv*sequential_2/random_crop_2/cond/Cast_4:y:04sequential_2/random_crop_2/cond/truediv_2/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_5Cast-sequential_2/random_crop_2/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
%sequential_2/random_crop_2/cond/sub_1Sub8sequential_2/random_crop_2/cond/strided_slice_1:output:0-sequential_2/random_crop_2/cond/Minimum_1:z:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_6Cast)sequential_2/random_crop_2/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/random_crop_2/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
)sequential_2/random_crop_2/cond/truediv_3RealDiv*sequential_2/random_crop_2/cond/Cast_6:y:04sequential_2/random_crop_2/cond/truediv_3/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_7Cast-sequential_2/random_crop_2/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: i
'sequential_2/random_crop_2/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : i
'sequential_2/random_crop_2/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_2/random_crop_2/cond/stackPack0sequential_2/random_crop_2/cond/stack/0:output:0*sequential_2/random_crop_2/cond/Cast_5:y:0*sequential_2/random_crop_2/cond/Cast_7:y:00sequential_2/random_crop_2/cond/stack/3:output:0*
N*
T0*
_output_shapes
:t
)sequential_2/random_crop_2/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????t
)sequential_2/random_crop_2/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_2/random_crop_2/cond/stack_1Pack2sequential_2/random_crop_2/cond/stack_1/0:output:0+sequential_2/random_crop_2/cond/Minimum:z:0-sequential_2/random_crop_2/cond/Minimum_1:z:02sequential_2/random_crop_2/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
%sequential_2/random_crop_2/cond/SliceSliceNsequential_2_random_crop_2_cond_shape_sequential_2_center_crop_3_cond_identity.sequential_2/random_crop_2/cond/stack:output:00sequential_2/random_crop_2/cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????|
+sequential_2/random_crop_2/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
5sequential_2/random_crop_2/cond/resize/ResizeBilinearResizeBilinear.sequential_2/random_crop_2/cond/Slice:output:04sequential_2/random_crop_2/cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
(sequential_2/random_crop_2/cond/IdentityIdentityFsequential_2/random_crop_2/cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"]
(sequential_2_random_crop_2_cond_identity1sequential_2/random_crop_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?'
?	
B__inference_model_3_layer_call_and_return_conditional_losses_14431
input_8!
tf_nn_bias_add_3_biasadd_bias 
sequential_2_14366:	 
sequential_2_14368:	%
vgg16_14371:@
vgg16_14373:@%
vgg16_14375:@@
vgg16_14377:@&
vgg16_14379:@?
vgg16_14381:	?'
vgg16_14383:??
vgg16_14385:	?'
vgg16_14387:??
vgg16_14389:	?'
vgg16_14391:??
vgg16_14393:	?'
vgg16_14395:??
vgg16_14397:	?'
vgg16_14399:??
vgg16_14401:	?'
vgg16_14403:??
vgg16_14405:	?'
vgg16_14407:??
vgg16_14409:	?'
vgg16_14411:??
vgg16_14413:	?'
vgg16_14415:??
vgg16_14417:	?'
vgg16_14419:??
vgg16_14421:	? 
dense_3_14425:	?
dense_3_14427:
identity??dense_3/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?vgg16/StatefulPartitionedCall
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ?????
(tf.__operators__.getitem_3/strided_sliceStridedSliceinput_87tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask?
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:????????????
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall!tf.nn.bias_add_3/BiasAdd:output:0sequential_2_14366sequential_2_14368*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12931?
vgg16/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0vgg16_14371vgg16_14373vgg16_14375vgg16_14377vgg16_14379vgg16_14381vgg16_14383vgg16_14385vgg16_14387vgg16_14389vgg16_14391vgg16_14393vgg16_14395vgg16_14397vgg16_14399vgg16_14401vgg16_14403vgg16_14405vgg16_14407vgg16_14409vgg16_14411vgg16_14413vgg16_14415vgg16_14417vgg16_14419vgg16_14421*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_13261?
*global_average_pooling2d_3/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_13849?
dense_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0dense_3_14425dense_3_14427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_13932w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_3/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_8: 

_output_shapes
:
?~
?
random_crop_2_cond_true_15968I
;random_crop_2_cond_stateful_uniform_rngreadandskip_resource:	,
(random_crop_2_cond_add_random_crop_2_sub0
,random_crop_2_cond_add_1_random_crop_2_sub_1M
Irandom_crop_2_cond_crop_to_bounding_box_shape_center_crop_3_cond_identity
random_crop_2_cond_identity??5random_crop_2/cond/crop_to_bounding_box/Assert/Assert?7random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert?7random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert?7random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert?2random_crop_2/cond/stateful_uniform/RngReadAndSkips
)random_crop_2/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
'random_crop_2/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : m
'random_crop_2/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????s
)random_crop_2/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
(random_crop_2/cond/stateful_uniform/ProdProd2random_crop_2/cond/stateful_uniform/shape:output:02random_crop_2/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: l
*random_crop_2/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
*random_crop_2/cond/stateful_uniform/Cast_1Cast1random_crop_2/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
2random_crop_2/cond/stateful_uniform/RngReadAndSkipRngReadAndSkip;random_crop_2_cond_stateful_uniform_rngreadandskip_resource3random_crop_2/cond/stateful_uniform/Cast/x:output:0.random_crop_2/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:?
7random_crop_2/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9random_crop_2/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9random_crop_2/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1random_crop_2/cond/stateful_uniform/strided_sliceStridedSlice:random_crop_2/cond/stateful_uniform/RngReadAndSkip:value:0@random_crop_2/cond/stateful_uniform/strided_slice/stack:output:0Brandom_crop_2/cond/stateful_uniform/strided_slice/stack_1:output:0Brandom_crop_2/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
+random_crop_2/cond/stateful_uniform/BitcastBitcast:random_crop_2/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
9random_crop_2/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
;random_crop_2/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;random_crop_2/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3random_crop_2/cond/stateful_uniform/strided_slice_1StridedSlice:random_crop_2/cond/stateful_uniform/RngReadAndSkip:value:0Brandom_crop_2/cond/stateful_uniform/strided_slice_1/stack:output:0Drandom_crop_2/cond/stateful_uniform/strided_slice_1/stack_1:output:0Drandom_crop_2/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
-random_crop_2/cond/stateful_uniform/Bitcast_1Bitcast<random_crop_2/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'random_crop_2/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
#random_crop_2/cond/stateful_uniformStatelessRandomUniformIntV22random_crop_2/cond/stateful_uniform/shape:output:06random_crop_2/cond/stateful_uniform/Bitcast_1:output:04random_crop_2/cond/stateful_uniform/Bitcast:output:00random_crop_2/cond/stateful_uniform/alg:output:00random_crop_2/cond/stateful_uniform/min:output:00random_crop_2/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0p
&random_crop_2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(random_crop_2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(random_crop_2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop_2/cond/strided_sliceStridedSlice,random_crop_2/cond/stateful_uniform:output:0/random_crop_2/cond/strided_slice/stack:output:01random_crop_2/cond/strided_slice/stack_1:output:01random_crop_2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
random_crop_2/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop_2/cond/addAddV2(random_crop_2_cond_add_random_crop_2_sub!random_crop_2/cond/add/y:output:0*
T0*
_output_shapes
: ?
random_crop_2/cond/modFloorMod)random_crop_2/cond/strided_slice:output:0random_crop_2/cond/add:z:0*
T0*
_output_shapes
: r
(random_crop_2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*random_crop_2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*random_crop_2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"random_crop_2/cond/strided_slice_1StridedSlice,random_crop_2/cond/stateful_uniform:output:01random_crop_2/cond/strided_slice_1/stack:output:03random_crop_2/cond/strided_slice_1/stack_1:output:03random_crop_2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
random_crop_2/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop_2/cond/add_1AddV2,random_crop_2_cond_add_1_random_crop_2_sub_1#random_crop_2/cond/add_1/y:output:0*
T0*
_output_shapes
: ?
random_crop_2/cond/mod_1FloorMod+random_crop_2/cond/strided_slice_1:output:0random_crop_2/cond/add_1:z:0*
T0*
_output_shapes
: ?
-random_crop_2/cond/crop_to_bounding_box/ShapeShapeIrandom_crop_2_cond_crop_to_bounding_box_shape_center_crop_3_cond_identity*
T0*
_output_shapes
:?
/random_crop_2/cond/crop_to_bounding_box/unstackUnpack6random_crop_2/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numx
6random_crop_2/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
4random_crop_2/cond/crop_to_bounding_box/GreaterEqualGreaterEqualrandom_crop_2/cond/mod_1:z:0?random_crop_2/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
4random_crop_2/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
<random_crop_2/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
5random_crop_2/cond/crop_to_bounding_box/Assert/AssertAssert8random_crop_2/cond/crop_to_bounding_box/GreaterEqual:z:0Erandom_crop_2/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 z
8random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
6random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualrandom_crop_2/cond/mod:z:0Arandom_crop_2/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
6random_crop_2/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
>random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
7random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertAssert:random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1:z:0Grandom_crop_2/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:06^random_crop_2/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 p
-random_crop_2/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
+random_crop_2/cond/crop_to_bounding_box/addAddV26random_crop_2/cond/crop_to_bounding_box/add/x:output:0random_crop_2/cond/mod_1:z:0*
T0*
_output_shapes
: v
3random_crop_2/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
1random_crop_2/cond/crop_to_bounding_box/LessEqual	LessEqual/random_crop_2/cond/crop_to_bounding_box/add:z:0<random_crop_2/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
6random_crop_2/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
>random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
7random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertAssert5random_crop_2/cond/crop_to_bounding_box/LessEqual:z:0Grandom_crop_2/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:08^random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 r
/random_crop_2/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
-random_crop_2/cond/crop_to_bounding_box/add_1AddV28random_crop_2/cond/crop_to_bounding_box/add_1/x:output:0random_crop_2/cond/mod:z:0*
T0*
_output_shapes
: x
5random_crop_2/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
3random_crop_2/cond/crop_to_bounding_box/LessEqual_1	LessEqual1random_crop_2/cond/crop_to_bounding_box/add_1:z:0>random_crop_2/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
6random_crop_2/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
>random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
7random_crop_2/cond/crop_to_bounding_box/Assert_3/AssertAssert7random_crop_2/cond/crop_to_bounding_box/LessEqual_1:z:0Grandom_crop_2/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:08^random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
:random_crop_2/cond/crop_to_bounding_box/control_dependencyIdentityIrandom_crop_2_cond_crop_to_bounding_box_shape_center_crop_3_cond_identity6^random_crop_2/cond/crop_to_bounding_box/Assert/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert*
T0*.
_class$
" loc:@center_crop_3/cond/Identity*1
_output_shapes
:???????????q
/random_crop_2/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : q
/random_crop_2/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
-random_crop_2/cond/crop_to_bounding_box/stackPack8random_crop_2/cond/crop_to_bounding_box/stack/0:output:0random_crop_2/cond/mod:z:0random_crop_2/cond/mod_1:z:08random_crop_2/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
/random_crop_2/cond/crop_to_bounding_box/Shape_1ShapeCrandom_crop_2/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
;random_crop_2/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5random_crop_2/cond/crop_to_bounding_box/strided_sliceStridedSlice8random_crop_2/cond/crop_to_bounding_box/Shape_1:output:0Drandom_crop_2/cond/crop_to_bounding_box/strided_slice/stack:output:0Frandom_crop_2/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Frandom_crop_2/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
/random_crop_2/cond/crop_to_bounding_box/Shape_2ShapeCrandom_crop_2/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
=random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
?random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7random_crop_2/cond/crop_to_bounding_box/strided_slice_1StridedSlice8random_crop_2/cond/crop_to_bounding_box/Shape_2:output:0Frandom_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Hrandom_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Hrandom_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
1random_crop_2/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :?t
1random_crop_2/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
/random_crop_2/cond/crop_to_bounding_box/stack_1Pack>random_crop_2/cond/crop_to_bounding_box/strided_slice:output:0:random_crop_2/cond/crop_to_bounding_box/stack_1/1:output:0:random_crop_2/cond/crop_to_bounding_box/stack_1/2:output:0@random_crop_2/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
-random_crop_2/cond/crop_to_bounding_box/SliceSliceCrandom_crop_2/cond/crop_to_bounding_box/control_dependency:output:06random_crop_2/cond/crop_to_bounding_box/stack:output:08random_crop_2/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
random_crop_2/cond/IdentityIdentity6random_crop_2/cond/crop_to_bounding_box/Slice:output:0^random_crop_2/cond/NoOp*
T0*1
_output_shapes
:????????????
random_crop_2/cond/NoOpNoOp6^random_crop_2/cond/crop_to_bounding_box/Assert/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert3^random_crop_2/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "C
random_crop_2_cond_identity$random_crop_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????2n
5random_crop_2/cond/crop_to_bounding_box/Assert/Assert5random_crop_2/cond/crop_to_bounding_box/Assert/Assert2r
7random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert7random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert2r
7random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert7random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert2r
7random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert7random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert2h
2random_crop_2/cond/stateful_uniform/RngReadAndSkip2random_crop_2/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
q
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_13849

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_13062

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?=
?
+sequential_2_center_crop_3_cond_false_15012/
+sequential_2_center_crop_3_cond_placeholder1
-sequential_2_center_crop_3_cond_placeholder_1B
>sequential_2_center_crop_3_cond_shape_tf_nn_bias_add_3_biasadd,
(sequential_2_center_crop_3_cond_identity?
%sequential_2/center_crop_3/cond/ShapeShape>sequential_2_center_crop_3_cond_shape_tf_nn_bias_add_3_biasadd*
T0*
_output_shapes
:?
3sequential_2/center_crop_3/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
5sequential_2/center_crop_3/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????
5sequential_2/center_crop_3/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential_2/center_crop_3/cond/strided_sliceStridedSlice.sequential_2/center_crop_3/cond/Shape:output:0<sequential_2/center_crop_3/cond/strided_slice/stack:output:0>sequential_2/center_crop_3/cond/strided_slice/stack_1:output:0>sequential_2/center_crop_3/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_2/center_crop_3/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
7sequential_2/center_crop_3/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
7sequential_2/center_crop_3/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_2/center_crop_3/cond/strided_slice_1StridedSlice.sequential_2/center_crop_3/cond/Shape:output:0>sequential_2/center_crop_3/cond/strided_slice_1/stack:output:0@sequential_2/center_crop_3/cond/strided_slice_1/stack_1:output:0@sequential_2/center_crop_3/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%sequential_2/center_crop_3/cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
#sequential_2/center_crop_3/cond/mulMul8sequential_2/center_crop_3/cond/strided_slice_1:output:0.sequential_2/center_crop_3/cond/mul/y:output:0*
T0*
_output_shapes
: ?
$sequential_2/center_crop_3/cond/CastCast'sequential_2/center_crop_3/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: n
)sequential_2/center_crop_3/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
'sequential_2/center_crop_3/cond/truedivRealDiv(sequential_2/center_crop_3/cond/Cast:y:02sequential_2/center_crop_3/cond/truediv/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_1Cast+sequential_2/center_crop_3/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: j
'sequential_2/center_crop_3/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
%sequential_2/center_crop_3/cond/mul_1Mul6sequential_2/center_crop_3/cond/strided_slice:output:00sequential_2/center_crop_3/cond/mul_1/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_2Cast)sequential_2/center_crop_3/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/center_crop_3/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
)sequential_2/center_crop_3/cond/truediv_1RealDiv*sequential_2/center_crop_3/cond/Cast_2:y:04sequential_2/center_crop_3/cond/truediv_1/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_3Cast-sequential_2/center_crop_3/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
'sequential_2/center_crop_3/cond/MinimumMinimum6sequential_2/center_crop_3/cond/strided_slice:output:0*sequential_2/center_crop_3/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
)sequential_2/center_crop_3/cond/Minimum_1Minimum8sequential_2/center_crop_3/cond/strided_slice_1:output:0*sequential_2/center_crop_3/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
#sequential_2/center_crop_3/cond/subSub6sequential_2/center_crop_3/cond/strided_slice:output:0+sequential_2/center_crop_3/cond/Minimum:z:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_4Cast'sequential_2/center_crop_3/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/center_crop_3/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
)sequential_2/center_crop_3/cond/truediv_2RealDiv*sequential_2/center_crop_3/cond/Cast_4:y:04sequential_2/center_crop_3/cond/truediv_2/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_5Cast-sequential_2/center_crop_3/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
%sequential_2/center_crop_3/cond/sub_1Sub8sequential_2/center_crop_3/cond/strided_slice_1:output:0-sequential_2/center_crop_3/cond/Minimum_1:z:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_6Cast)sequential_2/center_crop_3/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/center_crop_3/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
)sequential_2/center_crop_3/cond/truediv_3RealDiv*sequential_2/center_crop_3/cond/Cast_6:y:04sequential_2/center_crop_3/cond/truediv_3/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_7Cast-sequential_2/center_crop_3/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: i
'sequential_2/center_crop_3/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : i
'sequential_2/center_crop_3/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_2/center_crop_3/cond/stackPack0sequential_2/center_crop_3/cond/stack/0:output:0*sequential_2/center_crop_3/cond/Cast_5:y:0*sequential_2/center_crop_3/cond/Cast_7:y:00sequential_2/center_crop_3/cond/stack/3:output:0*
N*
T0*
_output_shapes
:t
)sequential_2/center_crop_3/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????t
)sequential_2/center_crop_3/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_2/center_crop_3/cond/stack_1Pack2sequential_2/center_crop_3/cond/stack_1/0:output:0+sequential_2/center_crop_3/cond/Minimum:z:0-sequential_2/center_crop_3/cond/Minimum_1:z:02sequential_2/center_crop_3/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
%sequential_2/center_crop_3/cond/SliceSlice>sequential_2_center_crop_3_cond_shape_tf_nn_bias_add_3_biasadd.sequential_2/center_crop_3/cond/stack:output:00sequential_2/center_crop_3/cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????|
+sequential_2/center_crop_3/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
5sequential_2/center_crop_3/cond/resize/ResizeBilinearResizeBilinear.sequential_2/center_crop_3/cond/Slice:output:04sequential_2/center_crop_3/cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
(sequential_2/center_crop_3/cond/IdentityIdentityFsequential_2/center_crop_3/cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"]
(sequential_2_center_crop_3_cond_identity1sequential_2/center_crop_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?Y
?
center_crop_3_cond_true_158335
1center_crop_3_cond_truediv_cast_center_crop_3_sub9
5center_crop_3_cond_truediv_1_cast_center_crop_3_sub_18
4center_crop_3_cond_crop_to_bounding_box_shape_inputs
center_crop_3_cond_identity??5center_crop_3/cond/crop_to_bounding_box/Assert/Assert?7center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert?7center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert?7center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert^
center_crop_3/cond/truediv/yConst*
_output_shapes
: *
dtype0*
value	B :?
center_crop_3/cond/truediv/CastCast1center_crop_3_cond_truediv_cast_center_crop_3_sub*

DstT0*

SrcT0*
_output_shapes
: ?
!center_crop_3/cond/truediv/Cast_1Cast%center_crop_3/cond/truediv/y:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
center_crop_3/cond/truedivRealDiv#center_crop_3/cond/truediv/Cast:y:0%center_crop_3/cond/truediv/Cast_1:y:0*
T0*
_output_shapes
: o
center_crop_3/cond/CastCastcenter_crop_3/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: `
center_crop_3/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!center_crop_3/cond/truediv_1/CastCast5center_crop_3_cond_truediv_1_cast_center_crop_3_sub_1*

DstT0*

SrcT0*
_output_shapes
: ?
#center_crop_3/cond/truediv_1/Cast_1Cast'center_crop_3/cond/truediv_1/y:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
center_crop_3/cond/truediv_1RealDiv%center_crop_3/cond/truediv_1/Cast:y:0'center_crop_3/cond/truediv_1/Cast_1:y:0*
T0*
_output_shapes
: s
center_crop_3/cond/Cast_1Cast center_crop_3/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
-center_crop_3/cond/crop_to_bounding_box/ShapeShape4center_crop_3_cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
/center_crop_3/cond/crop_to_bounding_box/unstackUnpack6center_crop_3/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numx
6center_crop_3/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
4center_crop_3/cond/crop_to_bounding_box/GreaterEqualGreaterEqualcenter_crop_3/cond/Cast_1:y:0?center_crop_3/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
4center_crop_3/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
<center_crop_3/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
5center_crop_3/cond/crop_to_bounding_box/Assert/AssertAssert8center_crop_3/cond/crop_to_bounding_box/GreaterEqual:z:0Ecenter_crop_3/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 z
8center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
6center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcenter_crop_3/cond/Cast:y:0Acenter_crop_3/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
6center_crop_3/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
>center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
7center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertAssert:center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1:z:0Gcenter_crop_3/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:06^center_crop_3/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 p
-center_crop_3/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
+center_crop_3/cond/crop_to_bounding_box/addAddV26center_crop_3/cond/crop_to_bounding_box/add/x:output:0center_crop_3/cond/Cast_1:y:0*
T0*
_output_shapes
: v
3center_crop_3/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
1center_crop_3/cond/crop_to_bounding_box/LessEqual	LessEqual/center_crop_3/cond/crop_to_bounding_box/add:z:0<center_crop_3/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
6center_crop_3/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
>center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
7center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertAssert5center_crop_3/cond/crop_to_bounding_box/LessEqual:z:0Gcenter_crop_3/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:08^center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 r
/center_crop_3/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
-center_crop_3/cond/crop_to_bounding_box/add_1AddV28center_crop_3/cond/crop_to_bounding_box/add_1/x:output:0center_crop_3/cond/Cast:y:0*
T0*
_output_shapes
: x
5center_crop_3/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
3center_crop_3/cond/crop_to_bounding_box/LessEqual_1	LessEqual1center_crop_3/cond/crop_to_bounding_box/add_1:z:0>center_crop_3/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
6center_crop_3/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
>center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
7center_crop_3/cond/crop_to_bounding_box/Assert_3/AssertAssert7center_crop_3/cond/crop_to_bounding_box/LessEqual_1:z:0Gcenter_crop_3/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:08^center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
:center_crop_3/cond/crop_to_bounding_box/control_dependencyIdentity4center_crop_3_cond_crop_to_bounding_box_shape_inputs6^center_crop_3/cond/crop_to_bounding_box/Assert/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*1
_output_shapes
:???????????q
/center_crop_3/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : q
/center_crop_3/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
-center_crop_3/cond/crop_to_bounding_box/stackPack8center_crop_3/cond/crop_to_bounding_box/stack/0:output:0center_crop_3/cond/Cast:y:0center_crop_3/cond/Cast_1:y:08center_crop_3/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
/center_crop_3/cond/crop_to_bounding_box/Shape_1ShapeCcenter_crop_3/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
;center_crop_3/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5center_crop_3/cond/crop_to_bounding_box/strided_sliceStridedSlice8center_crop_3/cond/crop_to_bounding_box/Shape_1:output:0Dcenter_crop_3/cond/crop_to_bounding_box/strided_slice/stack:output:0Fcenter_crop_3/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Fcenter_crop_3/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
/center_crop_3/cond/crop_to_bounding_box/Shape_2ShapeCcenter_crop_3/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
=center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
?center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7center_crop_3/cond/crop_to_bounding_box/strided_slice_1StridedSlice8center_crop_3/cond/crop_to_bounding_box/Shape_2:output:0Fcenter_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Hcenter_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Hcenter_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
1center_crop_3/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :?t
1center_crop_3/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
/center_crop_3/cond/crop_to_bounding_box/stack_1Pack>center_crop_3/cond/crop_to_bounding_box/strided_slice:output:0:center_crop_3/cond/crop_to_bounding_box/stack_1/1:output:0:center_crop_3/cond/crop_to_bounding_box/stack_1/2:output:0@center_crop_3/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
-center_crop_3/cond/crop_to_bounding_box/SliceSliceCcenter_crop_3/cond/crop_to_bounding_box/control_dependency:output:06center_crop_3/cond/crop_to_bounding_box/stack:output:08center_crop_3/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
center_crop_3/cond/IdentityIdentity6center_crop_3/cond/crop_to_bounding_box/Slice:output:0^center_crop_3/cond/NoOp*
T0*1
_output_shapes
:????????????
center_crop_3/cond/NoOpNoOp6^center_crop_3/cond/crop_to_bounding_box/Assert/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert*"
_acd_function_control_output(*
_output_shapes
 "C
center_crop_3_cond_identity$center_crop_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????2n
5center_crop_3/cond/crop_to_bounding_box/Assert/Assert5center_crop_3/cond/crop_to_bounding_box/Assert/Assert2r
7center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert7center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert2r
7center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert7center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert2r
7center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert7center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
G
+__inference_block4_pool_layer_call_fn_17266

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_13012?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block4_conv2_layer_call_and_return_conditional_losses_13184

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?[
?
__inference__traced_save_17516
file_prefix-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_random_crop_2_cond_statevar_read_readvariableop	9
5savev2_random_contrast_2_statevar_read_readvariableop	9
5savev2_rmsprop_dense_3_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_3_bias_rms_read_readvariableop>
:savev2_rmsprop_block5_conv1_kernel_rms_read_readvariableop<
8savev2_rmsprop_block5_conv1_bias_rms_read_readvariableop>
:savev2_rmsprop_block5_conv2_kernel_rms_read_readvariableop<
8savev2_rmsprop_block5_conv2_bias_rms_read_readvariableop>
:savev2_rmsprop_block5_conv3_kernel_rms_read_readvariableop<
8savev2_rmsprop_block5_conv3_bias_rms_read_readvariableop
savev2_const_1

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*?
value?B?0B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-3/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-3/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/22/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/23/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/24/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/25/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_random_crop_2_cond_statevar_read_readvariableop5savev2_random_contrast_2_statevar_read_readvariableop5savev2_rmsprop_dense_3_kernel_rms_read_readvariableop3savev2_rmsprop_dense_3_bias_rms_read_readvariableop:savev2_rmsprop_block5_conv1_kernel_rms_read_readvariableop8savev2_rmsprop_block5_conv1_bias_rms_read_readvariableop:savev2_rmsprop_block5_conv2_kernel_rms_read_readvariableop8savev2_rmsprop_block5_conv2_bias_rms_read_readvariableop:savev2_rmsprop_block5_conv3_kernel_rms_read_readvariableop8savev2_rmsprop_block5_conv3_bias_rms_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *>
dtypes4
220			?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?:: : : : : :@:@:@@:@:@?:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?: : : : :::	?::??:?:??:?:??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 	

_output_shapes
:@:,
(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:. *
(
_output_shapes
:??:!!

_output_shapes	
:?:"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: : &

_output_shapes
:: '

_output_shapes
::%(!

_output_shapes
:	?: )

_output_shapes
::.**
(
_output_shapes
:??:!+

_output_shapes	
:?:.,*
(
_output_shapes
:??:!-

_output_shapes	
:?:..*
(
_output_shapes
:??:!/

_output_shapes	
:?:0

_output_shapes
: 
?
}
-__inference_random_crop_2_layer_call_fn_16635

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_random_crop_2_layer_call_and_return_conditional_losses_12654y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
ޓ
?
*sequential_2_random_crop_2_cond_true_15146V
Hsequential_2_random_crop_2_cond_stateful_uniform_rngreadandskip_resource:	F
Bsequential_2_random_crop_2_cond_add_sequential_2_random_crop_2_subJ
Fsequential_2_random_crop_2_cond_add_1_sequential_2_random_crop_2_sub_1g
csequential_2_random_crop_2_cond_crop_to_bounding_box_shape_sequential_2_center_crop_3_cond_identity,
(sequential_2_random_crop_2_cond_identity??Bsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert?Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert?Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert?Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert??sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip?
6sequential_2/random_crop_2/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
4sequential_2/random_crop_2/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : z
4sequential_2/random_crop_2/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
6sequential_2/random_crop_2/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
5sequential_2/random_crop_2/cond/stateful_uniform/ProdProd?sequential_2/random_crop_2/cond/stateful_uniform/shape:output:0?sequential_2/random_crop_2/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: y
7sequential_2/random_crop_2/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
7sequential_2/random_crop_2/cond/stateful_uniform/Cast_1Cast>sequential_2/random_crop_2/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
?sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkipRngReadAndSkipHsequential_2_random_crop_2_cond_stateful_uniform_rngreadandskip_resource@sequential_2/random_crop_2/cond/stateful_uniform/Cast/x:output:0;sequential_2/random_crop_2/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:?
Dsequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential_2/random_crop_2/cond/stateful_uniform/strided_sliceStridedSliceGsequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip:value:0Msequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack:output:0Osequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_1:output:0Osequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
8sequential_2/random_crop_2/cond/stateful_uniform/BitcastBitcastGsequential_2/random_crop_2/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
Fsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1StridedSliceGsequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip:value:0Osequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack:output:0Qsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_1:output:0Qsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
:sequential_2/random_crop_2/cond/stateful_uniform/Bitcast_1BitcastIsequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0v
4sequential_2/random_crop_2/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
0sequential_2/random_crop_2/cond/stateful_uniformStatelessRandomUniformIntV2?sequential_2/random_crop_2/cond/stateful_uniform/shape:output:0Csequential_2/random_crop_2/cond/stateful_uniform/Bitcast_1:output:0Asequential_2/random_crop_2/cond/stateful_uniform/Bitcast:output:0=sequential_2/random_crop_2/cond/stateful_uniform/alg:output:0=sequential_2/random_crop_2/cond/stateful_uniform/min:output:0=sequential_2/random_crop_2/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0}
3sequential_2/random_crop_2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_2/random_crop_2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/random_crop_2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential_2/random_crop_2/cond/strided_sliceStridedSlice9sequential_2/random_crop_2/cond/stateful_uniform:output:0<sequential_2/random_crop_2/cond/strided_slice/stack:output:0>sequential_2/random_crop_2/cond/strided_slice/stack_1:output:0>sequential_2/random_crop_2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%sequential_2/random_crop_2/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential_2/random_crop_2/cond/addAddV2Bsequential_2_random_crop_2_cond_add_sequential_2_random_crop_2_sub.sequential_2/random_crop_2/cond/add/y:output:0*
T0*
_output_shapes
: ?
#sequential_2/random_crop_2/cond/modFloorMod6sequential_2/random_crop_2/cond/strided_slice:output:0'sequential_2/random_crop_2/cond/add:z:0*
T0*
_output_shapes
: 
5sequential_2/random_crop_2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7sequential_2/random_crop_2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_2/random_crop_2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_2/random_crop_2/cond/strided_slice_1StridedSlice9sequential_2/random_crop_2/cond/stateful_uniform:output:0>sequential_2/random_crop_2/cond/strided_slice_1/stack:output:0@sequential_2/random_crop_2/cond/strided_slice_1/stack_1:output:0@sequential_2/random_crop_2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'sequential_2/random_crop_2/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
%sequential_2/random_crop_2/cond/add_1AddV2Fsequential_2_random_crop_2_cond_add_1_sequential_2_random_crop_2_sub_10sequential_2/random_crop_2/cond/add_1/y:output:0*
T0*
_output_shapes
: ?
%sequential_2/random_crop_2/cond/mod_1FloorMod8sequential_2/random_crop_2/cond/strided_slice_1:output:0)sequential_2/random_crop_2/cond/add_1:z:0*
T0*
_output_shapes
: ?
:sequential_2/random_crop_2/cond/crop_to_bounding_box/ShapeShapecsequential_2_random_crop_2_cond_crop_to_bounding_box_shape_sequential_2_center_crop_3_cond_identity*
T0*
_output_shapes
:?
<sequential_2/random_crop_2/cond/crop_to_bounding_box/unstackUnpackCsequential_2/random_crop_2/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num?
Csequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Asequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqualGreaterEqual)sequential_2/random_crop_2/cond/mod_1:z:0Lsequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Asequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Isequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Bsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertAssertEsequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual:z:0Rsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
Esequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Csequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual'sequential_2/random_crop_2/cond/mod:z:0Nsequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Ksequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertAssertGsequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1:z:0Tsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0C^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 }
:sequential_2/random_crop_2/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
8sequential_2/random_crop_2/cond/crop_to_bounding_box/addAddV2Csequential_2/random_crop_2/cond/crop_to_bounding_box/add/x:output:0)sequential_2/random_crop_2/cond/mod_1:z:0*
T0*
_output_shapes
: ?
@sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
>sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual	LessEqual<sequential_2/random_crop_2/cond/crop_to_bounding_box/add:z:0Isequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Ksequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertAssertBsequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual:z:0Tsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0E^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 
<sequential_2/random_crop_2/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
:sequential_2/random_crop_2/cond/crop_to_bounding_box/add_1AddV2Esequential_2/random_crop_2/cond/crop_to_bounding_box/add_1/x:output:0'sequential_2/random_crop_2/cond/mod:z:0*
T0*
_output_shapes
: ?
Bsequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
@sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1	LessEqual>sequential_2/random_crop_2/cond/crop_to_bounding_box/add_1:z:0Ksequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Ksequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/AssertAssertDsequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1:z:0Tsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0E^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
Gsequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependencyIdentitycsequential_2_random_crop_2_cond_crop_to_bounding_box_shape_sequential_2_center_crop_3_cond_identityC^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert*
T0*;
_class1
/-loc:@sequential_2/center_crop_3/cond/Identity*1
_output_shapes
:???????????~
<sequential_2/random_crop_2/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : ~
<sequential_2/random_crop_2/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
:sequential_2/random_crop_2/cond/crop_to_bounding_box/stackPackEsequential_2/random_crop_2/cond/crop_to_bounding_box/stack/0:output:0'sequential_2/random_crop_2/cond/mod:z:0)sequential_2/random_crop_2/cond/mod_1:z:0Esequential_2/random_crop_2/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
<sequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_1ShapePsequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Hsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Jsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Jsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_sliceStridedSliceEsequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_1:output:0Qsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack:output:0Ssequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Ssequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<sequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_2ShapePsequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Jsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Lsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Lsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1StridedSliceEsequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_2:output:0Ssequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Usequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Usequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :??
>sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
<sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1PackKsequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice:output:0Gsequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/1:output:0Gsequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/2:output:0Msequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
:sequential_2/random_crop_2/cond/crop_to_bounding_box/SliceSlicePsequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependency:output:0Csequential_2/random_crop_2/cond/crop_to_bounding_box/stack:output:0Esequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
(sequential_2/random_crop_2/cond/IdentityIdentityCsequential_2/random_crop_2/cond/crop_to_bounding_box/Slice:output:0%^sequential_2/random_crop_2/cond/NoOp*
T0*1
_output_shapes
:????????????
$sequential_2/random_crop_2/cond/NoOpNoOpC^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertE^sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert@^sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_2_random_crop_2_cond_identity1sequential_2/random_crop_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????2?
Bsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertBsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert2?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertDsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert2?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertDsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert2?
Dsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/AssertDsequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert2?
?sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip?sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?=
?
+sequential_2_random_crop_2_cond_false_15147/
+sequential_2_random_crop_2_cond_placeholder1
-sequential_2_random_crop_2_cond_placeholder_11
-sequential_2_random_crop_2_cond_placeholder_2R
Nsequential_2_random_crop_2_cond_shape_sequential_2_center_crop_3_cond_identity,
(sequential_2_random_crop_2_cond_identity?
%sequential_2/random_crop_2/cond/ShapeShapeNsequential_2_random_crop_2_cond_shape_sequential_2_center_crop_3_cond_identity*
T0*
_output_shapes
:?
3sequential_2/random_crop_2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
5sequential_2/random_crop_2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????
5sequential_2/random_crop_2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential_2/random_crop_2/cond/strided_sliceStridedSlice.sequential_2/random_crop_2/cond/Shape:output:0<sequential_2/random_crop_2/cond/strided_slice/stack:output:0>sequential_2/random_crop_2/cond/strided_slice/stack_1:output:0>sequential_2/random_crop_2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_2/random_crop_2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
7sequential_2/random_crop_2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
7sequential_2/random_crop_2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_2/random_crop_2/cond/strided_slice_1StridedSlice.sequential_2/random_crop_2/cond/Shape:output:0>sequential_2/random_crop_2/cond/strided_slice_1/stack:output:0@sequential_2/random_crop_2/cond/strided_slice_1/stack_1:output:0@sequential_2/random_crop_2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%sequential_2/random_crop_2/cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
#sequential_2/random_crop_2/cond/mulMul8sequential_2/random_crop_2/cond/strided_slice_1:output:0.sequential_2/random_crop_2/cond/mul/y:output:0*
T0*
_output_shapes
: ?
$sequential_2/random_crop_2/cond/CastCast'sequential_2/random_crop_2/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: n
)sequential_2/random_crop_2/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
'sequential_2/random_crop_2/cond/truedivRealDiv(sequential_2/random_crop_2/cond/Cast:y:02sequential_2/random_crop_2/cond/truediv/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_1Cast+sequential_2/random_crop_2/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: j
'sequential_2/random_crop_2/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
%sequential_2/random_crop_2/cond/mul_1Mul6sequential_2/random_crop_2/cond/strided_slice:output:00sequential_2/random_crop_2/cond/mul_1/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_2Cast)sequential_2/random_crop_2/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/random_crop_2/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
)sequential_2/random_crop_2/cond/truediv_1RealDiv*sequential_2/random_crop_2/cond/Cast_2:y:04sequential_2/random_crop_2/cond/truediv_1/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_3Cast-sequential_2/random_crop_2/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
'sequential_2/random_crop_2/cond/MinimumMinimum6sequential_2/random_crop_2/cond/strided_slice:output:0*sequential_2/random_crop_2/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
)sequential_2/random_crop_2/cond/Minimum_1Minimum8sequential_2/random_crop_2/cond/strided_slice_1:output:0*sequential_2/random_crop_2/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
#sequential_2/random_crop_2/cond/subSub6sequential_2/random_crop_2/cond/strided_slice:output:0+sequential_2/random_crop_2/cond/Minimum:z:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_4Cast'sequential_2/random_crop_2/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/random_crop_2/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
)sequential_2/random_crop_2/cond/truediv_2RealDiv*sequential_2/random_crop_2/cond/Cast_4:y:04sequential_2/random_crop_2/cond/truediv_2/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_5Cast-sequential_2/random_crop_2/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
%sequential_2/random_crop_2/cond/sub_1Sub8sequential_2/random_crop_2/cond/strided_slice_1:output:0-sequential_2/random_crop_2/cond/Minimum_1:z:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_6Cast)sequential_2/random_crop_2/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/random_crop_2/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
)sequential_2/random_crop_2/cond/truediv_3RealDiv*sequential_2/random_crop_2/cond/Cast_6:y:04sequential_2/random_crop_2/cond/truediv_3/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/random_crop_2/cond/Cast_7Cast-sequential_2/random_crop_2/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: i
'sequential_2/random_crop_2/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : i
'sequential_2/random_crop_2/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_2/random_crop_2/cond/stackPack0sequential_2/random_crop_2/cond/stack/0:output:0*sequential_2/random_crop_2/cond/Cast_5:y:0*sequential_2/random_crop_2/cond/Cast_7:y:00sequential_2/random_crop_2/cond/stack/3:output:0*
N*
T0*
_output_shapes
:t
)sequential_2/random_crop_2/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????t
)sequential_2/random_crop_2/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_2/random_crop_2/cond/stack_1Pack2sequential_2/random_crop_2/cond/stack_1/0:output:0+sequential_2/random_crop_2/cond/Minimum:z:0-sequential_2/random_crop_2/cond/Minimum_1:z:02sequential_2/random_crop_2/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
%sequential_2/random_crop_2/cond/SliceSliceNsequential_2_random_crop_2_cond_shape_sequential_2_center_crop_3_cond_identity.sequential_2/random_crop_2/cond/stack:output:00sequential_2/random_crop_2/cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????|
+sequential_2/random_crop_2/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
5sequential_2/random_crop_2/cond/resize/ResizeBilinearResizeBilinear.sequential_2/random_crop_2/cond/Slice:output:04sequential_2/random_crop_2/cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
(sequential_2/random_crop_2/cond/IdentityIdentityFsequential_2/random_crop_2/cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"]
(sequential_2_random_crop_2_cond_identity1sequential_2/random_crop_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?~
?
random_crop_2_cond_true_15669I
;random_crop_2_cond_stateful_uniform_rngreadandskip_resource:	,
(random_crop_2_cond_add_random_crop_2_sub0
,random_crop_2_cond_add_1_random_crop_2_sub_1M
Irandom_crop_2_cond_crop_to_bounding_box_shape_center_crop_3_cond_identity
random_crop_2_cond_identity??5random_crop_2/cond/crop_to_bounding_box/Assert/Assert?7random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert?7random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert?7random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert?2random_crop_2/cond/stateful_uniform/RngReadAndSkips
)random_crop_2/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
'random_crop_2/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : m
'random_crop_2/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????s
)random_crop_2/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
(random_crop_2/cond/stateful_uniform/ProdProd2random_crop_2/cond/stateful_uniform/shape:output:02random_crop_2/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: l
*random_crop_2/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
*random_crop_2/cond/stateful_uniform/Cast_1Cast1random_crop_2/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
2random_crop_2/cond/stateful_uniform/RngReadAndSkipRngReadAndSkip;random_crop_2_cond_stateful_uniform_rngreadandskip_resource3random_crop_2/cond/stateful_uniform/Cast/x:output:0.random_crop_2/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:?
7random_crop_2/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9random_crop_2/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9random_crop_2/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1random_crop_2/cond/stateful_uniform/strided_sliceStridedSlice:random_crop_2/cond/stateful_uniform/RngReadAndSkip:value:0@random_crop_2/cond/stateful_uniform/strided_slice/stack:output:0Brandom_crop_2/cond/stateful_uniform/strided_slice/stack_1:output:0Brandom_crop_2/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
+random_crop_2/cond/stateful_uniform/BitcastBitcast:random_crop_2/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
9random_crop_2/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
;random_crop_2/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;random_crop_2/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3random_crop_2/cond/stateful_uniform/strided_slice_1StridedSlice:random_crop_2/cond/stateful_uniform/RngReadAndSkip:value:0Brandom_crop_2/cond/stateful_uniform/strided_slice_1/stack:output:0Drandom_crop_2/cond/stateful_uniform/strided_slice_1/stack_1:output:0Drandom_crop_2/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
-random_crop_2/cond/stateful_uniform/Bitcast_1Bitcast<random_crop_2/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'random_crop_2/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
#random_crop_2/cond/stateful_uniformStatelessRandomUniformIntV22random_crop_2/cond/stateful_uniform/shape:output:06random_crop_2/cond/stateful_uniform/Bitcast_1:output:04random_crop_2/cond/stateful_uniform/Bitcast:output:00random_crop_2/cond/stateful_uniform/alg:output:00random_crop_2/cond/stateful_uniform/min:output:00random_crop_2/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0p
&random_crop_2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(random_crop_2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(random_crop_2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop_2/cond/strided_sliceStridedSlice,random_crop_2/cond/stateful_uniform:output:0/random_crop_2/cond/strided_slice/stack:output:01random_crop_2/cond/strided_slice/stack_1:output:01random_crop_2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
random_crop_2/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop_2/cond/addAddV2(random_crop_2_cond_add_random_crop_2_sub!random_crop_2/cond/add/y:output:0*
T0*
_output_shapes
: ?
random_crop_2/cond/modFloorMod)random_crop_2/cond/strided_slice:output:0random_crop_2/cond/add:z:0*
T0*
_output_shapes
: r
(random_crop_2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*random_crop_2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*random_crop_2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"random_crop_2/cond/strided_slice_1StridedSlice,random_crop_2/cond/stateful_uniform:output:01random_crop_2/cond/strided_slice_1/stack:output:03random_crop_2/cond/strided_slice_1/stack_1:output:03random_crop_2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
random_crop_2/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop_2/cond/add_1AddV2,random_crop_2_cond_add_1_random_crop_2_sub_1#random_crop_2/cond/add_1/y:output:0*
T0*
_output_shapes
: ?
random_crop_2/cond/mod_1FloorMod+random_crop_2/cond/strided_slice_1:output:0random_crop_2/cond/add_1:z:0*
T0*
_output_shapes
: ?
-random_crop_2/cond/crop_to_bounding_box/ShapeShapeIrandom_crop_2_cond_crop_to_bounding_box_shape_center_crop_3_cond_identity*
T0*
_output_shapes
:?
/random_crop_2/cond/crop_to_bounding_box/unstackUnpack6random_crop_2/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numx
6random_crop_2/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
4random_crop_2/cond/crop_to_bounding_box/GreaterEqualGreaterEqualrandom_crop_2/cond/mod_1:z:0?random_crop_2/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
4random_crop_2/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
<random_crop_2/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
5random_crop_2/cond/crop_to_bounding_box/Assert/AssertAssert8random_crop_2/cond/crop_to_bounding_box/GreaterEqual:z:0Erandom_crop_2/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 z
8random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
6random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualrandom_crop_2/cond/mod:z:0Arandom_crop_2/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
6random_crop_2/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
>random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
7random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertAssert:random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1:z:0Grandom_crop_2/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:06^random_crop_2/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 p
-random_crop_2/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
+random_crop_2/cond/crop_to_bounding_box/addAddV26random_crop_2/cond/crop_to_bounding_box/add/x:output:0random_crop_2/cond/mod_1:z:0*
T0*
_output_shapes
: v
3random_crop_2/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
1random_crop_2/cond/crop_to_bounding_box/LessEqual	LessEqual/random_crop_2/cond/crop_to_bounding_box/add:z:0<random_crop_2/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
6random_crop_2/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
>random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
7random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertAssert5random_crop_2/cond/crop_to_bounding_box/LessEqual:z:0Grandom_crop_2/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:08^random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 r
/random_crop_2/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
-random_crop_2/cond/crop_to_bounding_box/add_1AddV28random_crop_2/cond/crop_to_bounding_box/add_1/x:output:0random_crop_2/cond/mod:z:0*
T0*
_output_shapes
: x
5random_crop_2/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
3random_crop_2/cond/crop_to_bounding_box/LessEqual_1	LessEqual1random_crop_2/cond/crop_to_bounding_box/add_1:z:0>random_crop_2/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
6random_crop_2/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
>random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
7random_crop_2/cond/crop_to_bounding_box/Assert_3/AssertAssert7random_crop_2/cond/crop_to_bounding_box/LessEqual_1:z:0Grandom_crop_2/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:08^random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
:random_crop_2/cond/crop_to_bounding_box/control_dependencyIdentityIrandom_crop_2_cond_crop_to_bounding_box_shape_center_crop_3_cond_identity6^random_crop_2/cond/crop_to_bounding_box/Assert/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert*
T0*.
_class$
" loc:@center_crop_3/cond/Identity*1
_output_shapes
:???????????q
/random_crop_2/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : q
/random_crop_2/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
-random_crop_2/cond/crop_to_bounding_box/stackPack8random_crop_2/cond/crop_to_bounding_box/stack/0:output:0random_crop_2/cond/mod:z:0random_crop_2/cond/mod_1:z:08random_crop_2/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
/random_crop_2/cond/crop_to_bounding_box/Shape_1ShapeCrandom_crop_2/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
;random_crop_2/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5random_crop_2/cond/crop_to_bounding_box/strided_sliceStridedSlice8random_crop_2/cond/crop_to_bounding_box/Shape_1:output:0Drandom_crop_2/cond/crop_to_bounding_box/strided_slice/stack:output:0Frandom_crop_2/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Frandom_crop_2/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
/random_crop_2/cond/crop_to_bounding_box/Shape_2ShapeCrandom_crop_2/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
=random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
?random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7random_crop_2/cond/crop_to_bounding_box/strided_slice_1StridedSlice8random_crop_2/cond/crop_to_bounding_box/Shape_2:output:0Frandom_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Hrandom_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Hrandom_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
1random_crop_2/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :?t
1random_crop_2/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
/random_crop_2/cond/crop_to_bounding_box/stack_1Pack>random_crop_2/cond/crop_to_bounding_box/strided_slice:output:0:random_crop_2/cond/crop_to_bounding_box/stack_1/1:output:0:random_crop_2/cond/crop_to_bounding_box/stack_1/2:output:0@random_crop_2/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
-random_crop_2/cond/crop_to_bounding_box/SliceSliceCrandom_crop_2/cond/crop_to_bounding_box/control_dependency:output:06random_crop_2/cond/crop_to_bounding_box/stack:output:08random_crop_2/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
random_crop_2/cond/IdentityIdentity6random_crop_2/cond/crop_to_bounding_box/Slice:output:0^random_crop_2/cond/NoOp*
T0*1
_output_shapes
:????????????
random_crop_2/cond/NoOpNoOp6^random_crop_2/cond/crop_to_bounding_box/Assert/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert8^random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert3^random_crop_2/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "C
random_crop_2_cond_identity$random_crop_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????2n
5random_crop_2/cond/crop_to_bounding_box/Assert/Assert5random_crop_2/cond/crop_to_bounding_box/Assert/Assert2r
7random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert7random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert2r
7random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert7random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert2r
7random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert7random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert2h
2random_crop_2/cond/stateful_uniform/RngReadAndSkip2random_crop_2/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
G__inference_block4_conv3_layer_call_and_return_conditional_losses_13201

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?V
?
@__inference_vgg16_layer_call_and_return_conditional_losses_13765
input_7,
block1_conv1_13694:@ 
block1_conv1_13696:@,
block1_conv2_13699:@@ 
block1_conv2_13701:@-
block2_conv1_13705:@?!
block2_conv1_13707:	?.
block2_conv2_13710:??!
block2_conv2_13712:	?.
block3_conv1_13716:??!
block3_conv1_13718:	?.
block3_conv2_13721:??!
block3_conv2_13723:	?.
block3_conv3_13726:??!
block3_conv3_13728:	?.
block4_conv1_13732:??!
block4_conv1_13734:	?.
block4_conv2_13737:??!
block4_conv2_13739:	?.
block4_conv3_13742:??!
block4_conv3_13744:	?.
block5_conv1_13748:??!
block5_conv1_13750:	?.
block5_conv2_13753:??!
block5_conv2_13755:	?.
block5_conv3_13758:??!
block5_conv3_13760:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_7block1_conv1_13694block1_conv1_13696*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_13045?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_13699block1_conv2_13701*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_13062?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_12976?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_13705block2_conv1_13707*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_13080?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_13710block2_conv2_13712*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_13097?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_12988?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_13716block3_conv1_13718*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_13115?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_13721block3_conv2_13723*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_13132?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_13726block3_conv3_13728*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_13149?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_13000?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_13732block4_conv1_13734*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_13167?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_13737block4_conv2_13739*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_13184?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_13742block4_conv3_13744*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_13201?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_13012?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_13748block5_conv1_13750*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_13219?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_13753block5_conv2_13755*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_13236?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_13758block5_conv3_13760*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_13253?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_13024|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_7
?
?
,__inference_block3_conv3_layer_call_fn_17180

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_13149x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????88?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
,__inference_block3_conv2_layer_call_fn_17160

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_13132x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????88?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?I
?
cond_true_12376
cond_truediv_cast_sub
cond_truediv_1_cast_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/AssertP
cond/truediv/yConst*
_output_shapes
: *
dtype0*
value	B :`
cond/truediv/CastCastcond_truediv_cast_sub*

DstT0*

SrcT0*
_output_shapes
: d
cond/truediv/Cast_1Castcond/truediv/y:output:0*

DstT0*

SrcT0*
_output_shapes
: h
cond/truedivRealDivcond/truediv/Cast:y:0cond/truediv/Cast_1:y:0*
T0*
_output_shapes
: S
	cond/CastCastcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: R
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
value	B :f
cond/truediv_1/CastCastcond_truediv_1_cast_sub_1*

DstT0*

SrcT0*
_output_shapes
: h
cond/truediv_1/Cast_1Castcond/truediv_1/y:output:0*

DstT0*

SrcT0*
_output_shapes
: n
cond/truediv_1RealDivcond/truediv_1/Cast:y:0cond/truediv_1/Cast_1:y:0*
T0*
_output_shapes
: W
cond/Cast_1Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: u
cond/crop_to_bounding_box/ShapeShape&cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/Cast_1:y:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/Cast:y:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 b
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: h
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 d
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/Cast:y:0*
T0*
_output_shapes
: j
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*1
_output_shapes
:???????????c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/Cast:y:0cond/Cast_1:y:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :?f
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*1
_output_shapes
:????????????
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?2
?
center_crop_3_cond_false_15535"
center_crop_3_cond_placeholder$
 center_crop_3_cond_placeholder_1#
center_crop_3_cond_shape_inputs
center_crop_3_cond_identityg
center_crop_3/cond/ShapeShapecenter_crop_3_cond_shape_inputs*
T0*
_output_shapes
:y
&center_crop_3/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
(center_crop_3/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(center_crop_3/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 center_crop_3/cond/strided_sliceStridedSlice!center_crop_3/cond/Shape:output:0/center_crop_3/cond/strided_slice/stack:output:01center_crop_3/cond/strided_slice/stack_1:output:01center_crop_3/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
(center_crop_3/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*center_crop_3/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*center_crop_3/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"center_crop_3/cond/strided_slice_1StridedSlice!center_crop_3/cond/Shape:output:01center_crop_3/cond/strided_slice_1/stack:output:03center_crop_3/cond/strided_slice_1/stack_1:output:03center_crop_3/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
center_crop_3/cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
center_crop_3/cond/mulMul+center_crop_3/cond/strided_slice_1:output:0!center_crop_3/cond/mul/y:output:0*
T0*
_output_shapes
: k
center_crop_3/cond/CastCastcenter_crop_3/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: a
center_crop_3/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
center_crop_3/cond/truedivRealDivcenter_crop_3/cond/Cast:y:0%center_crop_3/cond/truediv/y:output:0*
T0*
_output_shapes
: q
center_crop_3/cond/Cast_1Castcenter_crop_3/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: ]
center_crop_3/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
center_crop_3/cond/mul_1Mul)center_crop_3/cond/strided_slice:output:0#center_crop_3/cond/mul_1/y:output:0*
T0*
_output_shapes
: o
center_crop_3/cond/Cast_2Castcenter_crop_3/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
center_crop_3/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
center_crop_3/cond/truediv_1RealDivcenter_crop_3/cond/Cast_2:y:0'center_crop_3/cond/truediv_1/y:output:0*
T0*
_output_shapes
: s
center_crop_3/cond/Cast_3Cast center_crop_3/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
center_crop_3/cond/MinimumMinimum)center_crop_3/cond/strided_slice:output:0center_crop_3/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
center_crop_3/cond/Minimum_1Minimum+center_crop_3/cond/strided_slice_1:output:0center_crop_3/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
center_crop_3/cond/subSub)center_crop_3/cond/strided_slice:output:0center_crop_3/cond/Minimum:z:0*
T0*
_output_shapes
: m
center_crop_3/cond/Cast_4Castcenter_crop_3/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: c
center_crop_3/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
center_crop_3/cond/truediv_2RealDivcenter_crop_3/cond/Cast_4:y:0'center_crop_3/cond/truediv_2/y:output:0*
T0*
_output_shapes
: s
center_crop_3/cond/Cast_5Cast center_crop_3/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
center_crop_3/cond/sub_1Sub+center_crop_3/cond/strided_slice_1:output:0 center_crop_3/cond/Minimum_1:z:0*
T0*
_output_shapes
: o
center_crop_3/cond/Cast_6Castcenter_crop_3/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
center_crop_3/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
center_crop_3/cond/truediv_3RealDivcenter_crop_3/cond/Cast_6:y:0'center_crop_3/cond/truediv_3/y:output:0*
T0*
_output_shapes
: s
center_crop_3/cond/Cast_7Cast center_crop_3/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: \
center_crop_3/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : \
center_crop_3/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
center_crop_3/cond/stackPack#center_crop_3/cond/stack/0:output:0center_crop_3/cond/Cast_5:y:0center_crop_3/cond/Cast_7:y:0#center_crop_3/cond/stack/3:output:0*
N*
T0*
_output_shapes
:g
center_crop_3/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????g
center_crop_3/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
center_crop_3/cond/stack_1Pack%center_crop_3/cond/stack_1/0:output:0center_crop_3/cond/Minimum:z:0 center_crop_3/cond/Minimum_1:z:0%center_crop_3/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
center_crop_3/cond/SliceSlicecenter_crop_3_cond_shape_inputs!center_crop_3/cond/stack:output:0#center_crop_3/cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????o
center_crop_3/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
(center_crop_3/cond/resize/ResizeBilinearResizeBilinear!center_crop_3/cond/Slice:output:0'center_crop_3/cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
center_crop_3/cond/IdentityIdentity9center_crop_3/cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"C
center_crop_3_cond_identity$center_crop_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_17341

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_random_crop_2_layer_call_and_return_conditional_losses_12900

inputs
cond_input_0:	
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
sub/yConst*
_output_shapes
: *
dtype0*
value
B :?S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :?Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: O
Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z
Rank/packedPackRank/packed/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:M
All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z{
	All/inputPackAll/input/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_12764*0
output_shapes
:???????????*"
then_branchR
cond_true_12763d
cond/IdentityIdentitycond:output:0*
T0*1
_output_shapes
:???????????o
IdentityIdentitycond/Identity:output:0^NoOp*
T0*1
_output_shapes
:???????????M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2
condcond:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?D
?
3model_3_sequential_2_random_crop_2_cond_false_121027
3model_3_sequential_2_random_crop_2_cond_placeholder9
5model_3_sequential_2_random_crop_2_cond_placeholder_19
5model_3_sequential_2_random_crop_2_cond_placeholder_2b
^model_3_sequential_2_random_crop_2_cond_shape_model_3_sequential_2_center_crop_3_cond_identity4
0model_3_sequential_2_random_crop_2_cond_identity?
-model_3/sequential_2/random_crop_2/cond/ShapeShape^model_3_sequential_2_random_crop_2_cond_shape_model_3_sequential_2_center_crop_3_cond_identity*
T0*
_output_shapes
:?
;model_3/sequential_2/random_crop_2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
=model_3/sequential_2/random_crop_2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
=model_3/sequential_2/random_crop_2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5model_3/sequential_2/random_crop_2/cond/strided_sliceStridedSlice6model_3/sequential_2/random_crop_2/cond/Shape:output:0Dmodel_3/sequential_2/random_crop_2/cond/strided_slice/stack:output:0Fmodel_3/sequential_2/random_crop_2/cond/strided_slice/stack_1:output:0Fmodel_3/sequential_2/random_crop_2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
=model_3/sequential_2/random_crop_2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
?model_3/sequential_2/random_crop_2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
?model_3/sequential_2/random_crop_2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7model_3/sequential_2/random_crop_2/cond/strided_slice_1StridedSlice6model_3/sequential_2/random_crop_2/cond/Shape:output:0Fmodel_3/sequential_2/random_crop_2/cond/strided_slice_1/stack:output:0Hmodel_3/sequential_2/random_crop_2/cond/strided_slice_1/stack_1:output:0Hmodel_3/sequential_2/random_crop_2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
-model_3/sequential_2/random_crop_2/cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
+model_3/sequential_2/random_crop_2/cond/mulMul@model_3/sequential_2/random_crop_2/cond/strided_slice_1:output:06model_3/sequential_2/random_crop_2/cond/mul/y:output:0*
T0*
_output_shapes
: ?
,model_3/sequential_2/random_crop_2/cond/CastCast/model_3/sequential_2/random_crop_2/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: v
1model_3/sequential_2/random_crop_2/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
/model_3/sequential_2/random_crop_2/cond/truedivRealDiv0model_3/sequential_2/random_crop_2/cond/Cast:y:0:model_3/sequential_2/random_crop_2/cond/truediv/y:output:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/random_crop_2/cond/Cast_1Cast3model_3/sequential_2/random_crop_2/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: r
/model_3/sequential_2/random_crop_2/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
-model_3/sequential_2/random_crop_2/cond/mul_1Mul>model_3/sequential_2/random_crop_2/cond/strided_slice:output:08model_3/sequential_2/random_crop_2/cond/mul_1/y:output:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/random_crop_2/cond/Cast_2Cast1model_3/sequential_2/random_crop_2/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: x
3model_3/sequential_2/random_crop_2/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
1model_3/sequential_2/random_crop_2/cond/truediv_1RealDiv2model_3/sequential_2/random_crop_2/cond/Cast_2:y:0<model_3/sequential_2/random_crop_2/cond/truediv_1/y:output:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/random_crop_2/cond/Cast_3Cast5model_3/sequential_2/random_crop_2/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
/model_3/sequential_2/random_crop_2/cond/MinimumMinimum>model_3/sequential_2/random_crop_2/cond/strided_slice:output:02model_3/sequential_2/random_crop_2/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
1model_3/sequential_2/random_crop_2/cond/Minimum_1Minimum@model_3/sequential_2/random_crop_2/cond/strided_slice_1:output:02model_3/sequential_2/random_crop_2/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
+model_3/sequential_2/random_crop_2/cond/subSub>model_3/sequential_2/random_crop_2/cond/strided_slice:output:03model_3/sequential_2/random_crop_2/cond/Minimum:z:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/random_crop_2/cond/Cast_4Cast/model_3/sequential_2/random_crop_2/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: x
3model_3/sequential_2/random_crop_2/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
1model_3/sequential_2/random_crop_2/cond/truediv_2RealDiv2model_3/sequential_2/random_crop_2/cond/Cast_4:y:0<model_3/sequential_2/random_crop_2/cond/truediv_2/y:output:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/random_crop_2/cond/Cast_5Cast5model_3/sequential_2/random_crop_2/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
-model_3/sequential_2/random_crop_2/cond/sub_1Sub@model_3/sequential_2/random_crop_2/cond/strided_slice_1:output:05model_3/sequential_2/random_crop_2/cond/Minimum_1:z:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/random_crop_2/cond/Cast_6Cast1model_3/sequential_2/random_crop_2/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: x
3model_3/sequential_2/random_crop_2/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
1model_3/sequential_2/random_crop_2/cond/truediv_3RealDiv2model_3/sequential_2/random_crop_2/cond/Cast_6:y:0<model_3/sequential_2/random_crop_2/cond/truediv_3/y:output:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/random_crop_2/cond/Cast_7Cast5model_3/sequential_2/random_crop_2/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: q
/model_3/sequential_2/random_crop_2/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : q
/model_3/sequential_2/random_crop_2/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
-model_3/sequential_2/random_crop_2/cond/stackPack8model_3/sequential_2/random_crop_2/cond/stack/0:output:02model_3/sequential_2/random_crop_2/cond/Cast_5:y:02model_3/sequential_2/random_crop_2/cond/Cast_7:y:08model_3/sequential_2/random_crop_2/cond/stack/3:output:0*
N*
T0*
_output_shapes
:|
1model_3/sequential_2/random_crop_2/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????|
1model_3/sequential_2/random_crop_2/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
/model_3/sequential_2/random_crop_2/cond/stack_1Pack:model_3/sequential_2/random_crop_2/cond/stack_1/0:output:03model_3/sequential_2/random_crop_2/cond/Minimum:z:05model_3/sequential_2/random_crop_2/cond/Minimum_1:z:0:model_3/sequential_2/random_crop_2/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
-model_3/sequential_2/random_crop_2/cond/SliceSlice^model_3_sequential_2_random_crop_2_cond_shape_model_3_sequential_2_center_crop_3_cond_identity6model_3/sequential_2/random_crop_2/cond/stack:output:08model_3/sequential_2/random_crop_2/cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$?????????????????????
3model_3/sequential_2/random_crop_2/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
=model_3/sequential_2/random_crop_2/cond/resize/ResizeBilinearResizeBilinear6model_3/sequential_2/random_crop_2/cond/Slice:output:0<model_3/sequential_2/random_crop_2/cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
0model_3/sequential_2/random_crop_2/cond/IdentityIdentityNmodel_3/sequential_2/random_crop_2/cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"m
0model_3_sequential_2_random_crop_2_cond_identity9model_3/sequential_2/random_crop_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
M
1__inference_random_contrast_2_layer_call_fn_16979

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_12662j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_12670
center_crop_3_input
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcenter_crop_3_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12665y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:???????????
-
_user_specified_namecenter_crop_3_input
?
n
H__inference_center_crop_3_layer_call_and_return_conditional_losses_16628

inputs
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
sub/yConst*
_output_shapes
: *
dtype0*
value
B :?S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :?Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: g
Rank/packedPackGreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:e
	All/inputPackGreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_16519*0
output_shapes
:???????????*"
then_branchR
cond_true_16518d
cond/IdentityIdentitycond:output:0*
T0*1
_output_shapes
:???????????o
IdentityIdentitycond/Identity:output:0^NoOp*
T0*1
_output_shapes
:???????????M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????2
condcond:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_12947
center_crop_3_input
unknown:	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcenter_crop_3_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12931y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:???????????
-
_user_specified_namecenter_crop_3_input
?
?
,__inference_block2_conv2_layer_call_fn_17110

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_13097x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????pp?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????pp?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
??
?
@__inference_vgg16_layer_call_and_return_conditional_losses_16456

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@?;
,block2_conv1_biasadd_readvariableop_resource:	?G
+block2_conv2_conv2d_readvariableop_resource:??;
,block2_conv2_biasadd_readvariableop_resource:	?G
+block3_conv1_conv2d_readvariableop_resource:??;
,block3_conv1_biasadd_readvariableop_resource:	?G
+block3_conv2_conv2d_readvariableop_resource:??;
,block3_conv2_biasadd_readvariableop_resource:	?G
+block3_conv3_conv2d_readvariableop_resource:??;
,block3_conv3_biasadd_readvariableop_resource:	?G
+block4_conv1_conv2d_readvariableop_resource:??;
,block4_conv1_biasadd_readvariableop_resource:	?G
+block4_conv2_conv2d_readvariableop_resource:??;
,block4_conv2_biasadd_readvariableop_resource:	?G
+block4_conv3_conv2d_readvariableop_resource:??;
,block4_conv3_biasadd_readvariableop_resource:	?G
+block5_conv1_conv2d_readvariableop_resource:??;
,block5_conv1_biasadd_readvariableop_resource:	?G
+block5_conv2_conv2d_readvariableop_resource:??;
,block5_conv2_biasadd_readvariableop_resource:	?G
+block5_conv3_conv2d_readvariableop_resource:??;
,block5_conv3_biasadd_readvariableop_resource:	?
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?s
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp??
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?s
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp??
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block5_conv2_layer_call_and_return_conditional_losses_17311

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_17051

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block2_conv2_layer_call_and_return_conditional_losses_13097

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????pp?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????pp?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
?
?
'__inference_dense_3_layer_call_fn_16476

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_13932o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12967
center_crop_3_input!
random_crop_2_12960:	%
random_contrast_2_12963:	
identity??%center_crop_3/StatefulPartitionedCall?)random_contrast_2/StatefulPartitionedCall?%random_crop_2/StatefulPartitionedCall?
%center_crop_3/StatefulPartitionedCallStatefulPartitionedCallcenter_crop_3_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_center_crop_3_layer_call_and_return_conditional_losses_12486?
%random_crop_2/StatefulPartitionedCallStatefulPartitionedCall.center_crop_3/StatefulPartitionedCall:output:0random_crop_2_12960*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_random_crop_2_layer_call_and_return_conditional_losses_12900?
)random_contrast_2/StatefulPartitionedCallStatefulPartitionedCall.random_crop_2/StatefulPartitionedCall:output:0random_contrast_2_12963*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_12719?
IdentityIdentity2random_contrast_2/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp&^center_crop_3/StatefulPartitionedCall*^random_contrast_2/StatefulPartitionedCall&^random_crop_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 2N
%center_crop_3/StatefulPartitionedCall%center_crop_3/StatefulPartitionedCall2V
)random_contrast_2/StatefulPartitionedCall)random_contrast_2/StatefulPartitionedCall2N
%random_crop_2/StatefulPartitionedCall%random_crop_2/StatefulPartitionedCall:f b
1
_output_shapes
:???????????
-
_user_specified_namecenter_crop_3_input
?
h
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_12662

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
n
H__inference_center_crop_3_layer_call_and_return_conditional_losses_12486

inputs
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
sub/yConst*
_output_shapes
: *
dtype0*
value
B :?S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :?Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: g
Rank/packedPackGreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:e
	All/inputPackGreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_12377*0
output_shapes
:???????????*"
then_branchR
cond_true_12376d
cond/IdentityIdentitycond:output:0*
T0*1
_output_shapes
:???????????o
IdentityIdentitycond/Identity:output:0^NoOp*
T0*1
_output_shapes
:???????????M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????2
condcond:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
ڱ
?
B__inference_model_3_layer_call_and_return_conditional_losses_15424

inputs!
tf_nn_bias_add_3_biasadd_bias5
'sequential_2_random_crop_2_cond_input_0:	^
Psequential_2_random_contrast_2_stateful_uniform_full_int_rngreadandskip_resource:	K
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@?A
2vgg16_block2_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block2_conv2_conv2d_readvariableop_resource:??A
2vgg16_block2_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv1_conv2d_readvariableop_resource:??A
2vgg16_block3_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv2_conv2d_readvariableop_resource:??A
2vgg16_block3_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv3_conv2d_readvariableop_resource:??A
2vgg16_block3_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv1_conv2d_readvariableop_resource:??A
2vgg16_block4_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv2_conv2d_readvariableop_resource:??A
2vgg16_block4_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv3_conv2d_readvariableop_resource:??A
2vgg16_block4_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv1_conv2d_readvariableop_resource:??A
2vgg16_block5_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv2_conv2d_readvariableop_resource:??A
2vgg16_block5_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv3_conv2d_readvariableop_resource:??A
2vgg16_block5_conv3_biasadd_readvariableop_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?sequential_2/center_crop_3/cond?Gsequential_2/random_contrast_2/stateful_uniform_full_int/RngReadAndSkip?sequential_2/random_crop_2/cond?)vgg16/block1_conv1/BiasAdd/ReadVariableOp?(vgg16/block1_conv1/Conv2D/ReadVariableOp?)vgg16/block1_conv2/BiasAdd/ReadVariableOp?(vgg16/block1_conv2/Conv2D/ReadVariableOp?)vgg16/block2_conv1/BiasAdd/ReadVariableOp?(vgg16/block2_conv1/Conv2D/ReadVariableOp?)vgg16/block2_conv2/BiasAdd/ReadVariableOp?(vgg16/block2_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv1/BiasAdd/ReadVariableOp?(vgg16/block3_conv1/Conv2D/ReadVariableOp?)vgg16/block3_conv2/BiasAdd/ReadVariableOp?(vgg16/block3_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv3/BiasAdd/ReadVariableOp?(vgg16/block3_conv3/Conv2D/ReadVariableOp?)vgg16/block4_conv1/BiasAdd/ReadVariableOp?(vgg16/block4_conv1/Conv2D/ReadVariableOp?)vgg16/block4_conv2/BiasAdd/ReadVariableOp?(vgg16/block4_conv2/Conv2D/ReadVariableOp?)vgg16/block4_conv3/BiasAdd/ReadVariableOp?(vgg16/block4_conv3/Conv2D/ReadVariableOp?)vgg16/block5_conv1/BiasAdd/ReadVariableOp?(vgg16/block5_conv1/Conv2D/ReadVariableOp?)vgg16/block5_conv2/BiasAdd/ReadVariableOp?(vgg16/block5_conv2/Conv2D/ReadVariableOp?)vgg16/block5_conv3/BiasAdd/ReadVariableOp?(vgg16/block5_conv3/Conv2D/ReadVariableOp
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ?????
(tf.__operators__.getitem_3/strided_sliceStridedSliceinputs7tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask?
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:???????????q
 sequential_2/center_crop_3/ShapeShape!tf.nn.bias_add_3/BiasAdd:output:0*
T0*
_output_shapes
:?
.sequential_2/center_crop_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
0sequential_2/center_crop_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????z
0sequential_2/center_crop_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_2/center_crop_3/strided_sliceStridedSlice)sequential_2/center_crop_3/Shape:output:07sequential_2/center_crop_3/strided_slice/stack:output:09sequential_2/center_crop_3/strided_slice/stack_1:output:09sequential_2/center_crop_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 sequential_2/center_crop_3/sub/yConst*
_output_shapes
: *
dtype0*
value
B :??
sequential_2/center_crop_3/subSub1sequential_2/center_crop_3/strided_slice:output:0)sequential_2/center_crop_3/sub/y:output:0*
T0*
_output_shapes
: ?
0sequential_2/center_crop_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
2sequential_2/center_crop_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????|
2sequential_2/center_crop_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_2/center_crop_3/strided_slice_1StridedSlice)sequential_2/center_crop_3/Shape:output:09sequential_2/center_crop_3/strided_slice_1/stack:output:0;sequential_2/center_crop_3/strided_slice_1/stack_1:output:0;sequential_2/center_crop_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"sequential_2/center_crop_3/sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
 sequential_2/center_crop_3/sub_1Sub3sequential_2/center_crop_3/strided_slice_1:output:0+sequential_2/center_crop_3/sub_1/y:output:0*
T0*
_output_shapes
: k
)sequential_2/center_crop_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_2/center_crop_3/GreaterEqualGreaterEqual"sequential_2/center_crop_3/sub:z:02sequential_2/center_crop_3/GreaterEqual/y:output:0*
T0*
_output_shapes
: m
+sequential_2/center_crop_3/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_2/center_crop_3/GreaterEqual_1GreaterEqual$sequential_2/center_crop_3/sub_1:z:04sequential_2/center_crop_3/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/Rank/packedPack+sequential_2/center_crop_3/GreaterEqual:z:0-sequential_2/center_crop_3/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:a
sequential_2/center_crop_3/RankConst*
_output_shapes
: *
dtype0*
value	B :h
&sequential_2/center_crop_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : h
&sequential_2/center_crop_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
 sequential_2/center_crop_3/rangeRange/sequential_2/center_crop_3/range/start:output:0(sequential_2/center_crop_3/Rank:output:0/sequential_2/center_crop_3/range/delta:output:0*
_output_shapes
:?
$sequential_2/center_crop_3/All/inputPack+sequential_2/center_crop_3/GreaterEqual:z:0-sequential_2/center_crop_3/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:?
sequential_2/center_crop_3/AllAll-sequential_2/center_crop_3/All/input:output:0)sequential_2/center_crop_3/range:output:0*
_output_shapes
: ?
sequential_2/center_crop_3/condIf'sequential_2/center_crop_3/All:output:0"sequential_2/center_crop_3/sub:z:0$sequential_2/center_crop_3/sub_1:z:0!tf.nn.bias_add_3/BiasAdd:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *>
else_branch/R-
+sequential_2_center_crop_3_cond_false_15012*0
output_shapes
:???????????*=
then_branch.R,
*sequential_2_center_crop_3_cond_true_15011?
(sequential_2/center_crop_3/cond/IdentityIdentity(sequential_2/center_crop_3/cond:output:0*
T0*1
_output_shapes
:????????????
 sequential_2/random_crop_2/ShapeShape1sequential_2/center_crop_3/cond/Identity:output:0*
T0*
_output_shapes
:?
.sequential_2/random_crop_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
0sequential_2/random_crop_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????z
0sequential_2/random_crop_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_2/random_crop_2/strided_sliceStridedSlice)sequential_2/random_crop_2/Shape:output:07sequential_2/random_crop_2/strided_slice/stack:output:09sequential_2/random_crop_2/strided_slice/stack_1:output:09sequential_2/random_crop_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 sequential_2/random_crop_2/sub/yConst*
_output_shapes
: *
dtype0*
value
B :??
sequential_2/random_crop_2/subSub1sequential_2/random_crop_2/strided_slice:output:0)sequential_2/random_crop_2/sub/y:output:0*
T0*
_output_shapes
: ?
0sequential_2/random_crop_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
2sequential_2/random_crop_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????|
2sequential_2/random_crop_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_2/random_crop_2/strided_slice_1StridedSlice)sequential_2/random_crop_2/Shape:output:09sequential_2/random_crop_2/strided_slice_1/stack:output:0;sequential_2/random_crop_2/strided_slice_1/stack_1:output:0;sequential_2/random_crop_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"sequential_2/random_crop_2/sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
 sequential_2/random_crop_2/sub_1Sub3sequential_2/random_crop_2/strided_slice_1:output:0+sequential_2/random_crop_2/sub_1/y:output:0*
T0*
_output_shapes
: k
)sequential_2/random_crop_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_2/random_crop_2/GreaterEqualGreaterEqual"sequential_2/random_crop_2/sub:z:02sequential_2/random_crop_2/GreaterEqual/y:output:0*
T0*
_output_shapes
: m
+sequential_2/random_crop_2/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_2/random_crop_2/GreaterEqual_1GreaterEqual$sequential_2/random_crop_2/sub_1:z:04sequential_2/random_crop_2/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: j
(sequential_2/random_crop_2/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z?
&sequential_2/random_crop_2/Rank/packedPack1sequential_2/random_crop_2/Rank/packed/0:output:0+sequential_2/random_crop_2/GreaterEqual:z:0-sequential_2/random_crop_2/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:a
sequential_2/random_crop_2/RankConst*
_output_shapes
: *
dtype0*
value	B :h
&sequential_2/random_crop_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : h
&sequential_2/random_crop_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
 sequential_2/random_crop_2/rangeRange/sequential_2/random_crop_2/range/start:output:0(sequential_2/random_crop_2/Rank:output:0/sequential_2/random_crop_2/range/delta:output:0*
_output_shapes
:h
&sequential_2/random_crop_2/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z?
$sequential_2/random_crop_2/All/inputPack/sequential_2/random_crop_2/All/input/0:output:0+sequential_2/random_crop_2/GreaterEqual:z:0-sequential_2/random_crop_2/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:?
sequential_2/random_crop_2/AllAll-sequential_2/random_crop_2/All/input:output:0)sequential_2/random_crop_2/range:output:0*
_output_shapes
: ?
sequential_2/random_crop_2/condIf'sequential_2/random_crop_2/All:output:0'sequential_2_random_crop_2_cond_input_0"sequential_2/random_crop_2/sub:z:0$sequential_2/random_crop_2/sub_1:z:01sequential_2/center_crop_3/cond/Identity:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *>
else_branch/R-
+sequential_2_random_crop_2_cond_false_15147*0
output_shapes
:???????????*=
then_branch.R,
*sequential_2_random_crop_2_cond_true_15146?
(sequential_2/random_crop_2/cond/IdentityIdentity(sequential_2/random_crop_2/cond:output:0*
T0*1
_output_shapes
:????????????
>sequential_2/random_contrast_2/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
>sequential_2/random_contrast_2/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
=sequential_2/random_contrast_2/stateful_uniform_full_int/ProdProdGsequential_2/random_contrast_2/stateful_uniform_full_int/shape:output:0Gsequential_2/random_contrast_2/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: ?
?sequential_2/random_contrast_2/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
?sequential_2/random_contrast_2/stateful_uniform_full_int/Cast_1CastFsequential_2/random_contrast_2/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Gsequential_2/random_contrast_2/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipPsequential_2_random_contrast_2_stateful_uniform_full_int_rngreadandskip_resourceHsequential_2/random_contrast_2/stateful_uniform_full_int/Cast/x:output:0Csequential_2/random_contrast_2/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:?
Lsequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Nsequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Nsequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_2/random_contrast_2/stateful_uniform_full_int/strided_sliceStridedSliceOsequential_2/random_contrast_2/stateful_uniform_full_int/RngReadAndSkip:value:0Usequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice/stack:output:0Wsequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice/stack_1:output:0Wsequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
@sequential_2/random_contrast_2/stateful_uniform_full_int/BitcastBitcastOsequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
Nsequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Psequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hsequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice_1StridedSliceOsequential_2/random_contrast_2/stateful_uniform_full_int/RngReadAndSkip:value:0Wsequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice_1/stack:output:0Ysequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Ysequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
Bsequential_2/random_contrast_2/stateful_uniform_full_int/Bitcast_1BitcastQsequential_2/random_contrast_2/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0~
<sequential_2/random_contrast_2/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
8sequential_2/random_contrast_2/stateful_uniform_full_intStatelessRandomUniformFullIntV2Gsequential_2/random_contrast_2/stateful_uniform_full_int/shape:output:0Ksequential_2/random_contrast_2/stateful_uniform_full_int/Bitcast_1:output:0Isequential_2/random_contrast_2/stateful_uniform_full_int/Bitcast:output:0Esequential_2/random_contrast_2/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	s
)sequential_2/random_contrast_2/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ?
$sequential_2/random_contrast_2/stackPackAsequential_2/random_contrast_2/stateful_uniform_full_int:output:02sequential_2/random_contrast_2/zeros_like:output:0*
N*
T0	*
_output_shapes

:?
2sequential_2/random_contrast_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4sequential_2/random_contrast_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4sequential_2/random_contrast_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,sequential_2/random_contrast_2/strided_sliceStridedSlice-sequential_2/random_contrast_2/stack:output:0;sequential_2/random_contrast_2/strided_slice/stack:output:0=sequential_2/random_contrast_2/strided_slice/stack_1:output:0=sequential_2/random_contrast_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask?
=sequential_2/random_contrast_2/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB ?
;sequential_2/random_contrast_2/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
;sequential_2/random_contrast_2/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Tsequential_2/random_contrast_2/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter5sequential_2/random_contrast_2/strided_slice:output:0* 
_output_shapes
::?
Tsequential_2/random_contrast_2/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
Psequential_2/random_contrast_2/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Fsequential_2/random_contrast_2/stateless_random_uniform/shape:output:0Zsequential_2/random_contrast_2/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0^sequential_2/random_contrast_2/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0]sequential_2/random_contrast_2/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: ?
;sequential_2/random_contrast_2/stateless_random_uniform/subSubDsequential_2/random_contrast_2/stateless_random_uniform/max:output:0Dsequential_2/random_contrast_2/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
;sequential_2/random_contrast_2/stateless_random_uniform/mulMulYsequential_2/random_contrast_2/stateless_random_uniform/StatelessRandomUniformV2:output:0?sequential_2/random_contrast_2/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ?
7sequential_2/random_contrast_2/stateless_random_uniformAddV2?sequential_2/random_contrast_2/stateless_random_uniform/mul:z:0Dsequential_2/random_contrast_2/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
.sequential_2/random_contrast_2/adjust_contrastAdjustContrastv21sequential_2/random_crop_2/cond/Identity:output:0;sequential_2/random_contrast_2/stateless_random_uniform:z:0*1
_output_shapes
:????????????
7sequential_2/random_contrast_2/adjust_contrast/IdentityIdentity7sequential_2/random_contrast_2/adjust_contrast:output:0*
T0*1
_output_shapes
:????????????
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
vgg16/block1_conv1/Conv2DConv2D@sequential_2/random_contrast_2/adjust_contrast/Identity:output:00vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
?
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
?
)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp??
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
?
)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp??
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
?
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
1global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
global_average_pooling2d_3/MeanMean"vgg16/block5_pool/MaxPool:output:0:global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMul(global_average_pooling2d_3/Mean:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp ^sequential_2/center_crop_3/condH^sequential_2/random_contrast_2/stateful_uniform_full_int/RngReadAndSkip ^sequential_2/random_crop_2/cond*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2B
sequential_2/center_crop_3/condsequential_2/center_crop_3/cond2?
Gsequential_2/random_contrast_2/stateful_uniform_full_int/RngReadAndSkipGsequential_2/random_contrast_2/stateful_uniform_full_int/RngReadAndSkip2B
sequential_2/random_crop_2/condsequential_2/random_crop_2/cond2V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?
?
,__inference_block5_conv1_layer_call_fn_17280

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_13219x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
|
,__inference_sequential_2_layer_call_fn_15498

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12665y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_16467

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_vgg16_layer_call_fn_13691
input_7!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_13579x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_7
?
?
#__inference_signature_wrapper_15491
input_8
unknown
	unknown_0:	#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?

unknown_27:	?

unknown_28:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_12342o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_8: 

_output_shapes
:
?
?
H__inference_random_crop_2_layer_call_and_return_conditional_losses_16808

inputs
cond_input_0:	
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
sub/yConst*
_output_shapes
: *
dtype0*
value
B :?S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :?Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: O
Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z 
Rank/packedPackRank/packed/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:M
All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z {
	All/inputPackAll/input/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_16672*0
output_shapes
:???????????*"
then_branchR
cond_true_16671d
cond/IdentityIdentitycond:output:0*
T0*1
_output_shapes
:???????????o
IdentityIdentitycond/Identity:output:0^NoOp*
T0*1
_output_shapes
:???????????M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2
condcond:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_block4_conv3_layer_call_fn_17250

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_13201x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?x
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_16142

inputs(
random_crop_2_cond_input_0:	Q
Crandom_contrast_2_stateful_uniform_full_int_rngreadandskip_resource:	
identity??center_crop_3/cond?:random_contrast_2/stateful_uniform_full_int/RngReadAndSkip?random_crop_2/condI
center_crop_3/ShapeShapeinputs*
T0*
_output_shapes
:t
!center_crop_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
#center_crop_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#center_crop_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
center_crop_3/strided_sliceStridedSlicecenter_crop_3/Shape:output:0*center_crop_3/strided_slice/stack:output:0,center_crop_3/strided_slice/stack_1:output:0,center_crop_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
center_crop_3/sub/yConst*
_output_shapes
: *
dtype0*
value
B :?}
center_crop_3/subSub$center_crop_3/strided_slice:output:0center_crop_3/sub/y:output:0*
T0*
_output_shapes
: v
#center_crop_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????x
%center_crop_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%center_crop_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
center_crop_3/strided_slice_1StridedSlicecenter_crop_3/Shape:output:0,center_crop_3/strided_slice_1/stack:output:0.center_crop_3/strided_slice_1/stack_1:output:0.center_crop_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
center_crop_3/sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
center_crop_3/sub_1Sub&center_crop_3/strided_slice_1:output:0center_crop_3/sub_1/y:output:0*
T0*
_output_shapes
: ^
center_crop_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
center_crop_3/GreaterEqualGreaterEqualcenter_crop_3/sub:z:0%center_crop_3/GreaterEqual/y:output:0*
T0*
_output_shapes
: `
center_crop_3/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
center_crop_3/GreaterEqual_1GreaterEqualcenter_crop_3/sub_1:z:0'center_crop_3/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
center_crop_3/Rank/packedPackcenter_crop_3/GreaterEqual:z:0 center_crop_3/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:T
center_crop_3/RankConst*
_output_shapes
: *
dtype0*
value	B :[
center_crop_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : [
center_crop_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
center_crop_3/rangeRange"center_crop_3/range/start:output:0center_crop_3/Rank:output:0"center_crop_3/range/delta:output:0*
_output_shapes
:?
center_crop_3/All/inputPackcenter_crop_3/GreaterEqual:z:0 center_crop_3/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:p
center_crop_3/AllAll center_crop_3/All/input:output:0center_crop_3/range:output:0*
_output_shapes
: ?
center_crop_3/condIfcenter_crop_3/All:output:0center_crop_3/sub:z:0center_crop_3/sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *1
else_branch"R 
center_crop_3_cond_false_15834*0
output_shapes
:???????????*0
then_branch!R
center_crop_3_cond_true_15833?
center_crop_3/cond/IdentityIdentitycenter_crop_3/cond:output:0*
T0*1
_output_shapes
:???????????g
random_crop_2/ShapeShape$center_crop_3/cond/Identity:output:0*
T0*
_output_shapes
:t
!random_crop_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
#random_crop_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#random_crop_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop_2/strided_sliceStridedSlicerandom_crop_2/Shape:output:0*random_crop_2/strided_slice/stack:output:0,random_crop_2/strided_slice/stack_1:output:0,random_crop_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
random_crop_2/sub/yConst*
_output_shapes
: *
dtype0*
value
B :?}
random_crop_2/subSub$random_crop_2/strided_slice:output:0random_crop_2/sub/y:output:0*
T0*
_output_shapes
: v
#random_crop_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????x
%random_crop_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%random_crop_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop_2/strided_slice_1StridedSlicerandom_crop_2/Shape:output:0,random_crop_2/strided_slice_1/stack:output:0.random_crop_2/strided_slice_1/stack_1:output:0.random_crop_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
random_crop_2/sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
random_crop_2/sub_1Sub&random_crop_2/strided_slice_1:output:0random_crop_2/sub_1/y:output:0*
T0*
_output_shapes
: ^
random_crop_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop_2/GreaterEqualGreaterEqualrandom_crop_2/sub:z:0%random_crop_2/GreaterEqual/y:output:0*
T0*
_output_shapes
: `
random_crop_2/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop_2/GreaterEqual_1GreaterEqualrandom_crop_2/sub_1:z:0'random_crop_2/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ]
random_crop_2/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z?
random_crop_2/Rank/packedPack$random_crop_2/Rank/packed/0:output:0random_crop_2/GreaterEqual:z:0 random_crop_2/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:T
random_crop_2/RankConst*
_output_shapes
: *
dtype0*
value	B :[
random_crop_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : [
random_crop_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop_2/rangeRange"random_crop_2/range/start:output:0random_crop_2/Rank:output:0"random_crop_2/range/delta:output:0*
_output_shapes
:[
random_crop_2/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z?
random_crop_2/All/inputPack"random_crop_2/All/input/0:output:0random_crop_2/GreaterEqual:z:0 random_crop_2/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:p
random_crop_2/AllAll random_crop_2/All/input:output:0random_crop_2/range:output:0*
_output_shapes
: ?
random_crop_2/condIfrandom_crop_2/All:output:0random_crop_2_cond_input_0random_crop_2/sub:z:0random_crop_2/sub_1:z:0$center_crop_3/cond/Identity:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *1
else_branch"R 
random_crop_2_cond_false_15969*0
output_shapes
:???????????*0
then_branch!R
random_crop_2_cond_true_15968?
random_crop_2/cond/IdentityIdentityrandom_crop_2/cond:output:0*
T0*1
_output_shapes
:???????????{
1random_contrast_2/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:{
1random_contrast_2/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0random_contrast_2/stateful_uniform_full_int/ProdProd:random_contrast_2/stateful_uniform_full_int/shape:output:0:random_contrast_2/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: t
2random_contrast_2/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
2random_contrast_2/stateful_uniform_full_int/Cast_1Cast9random_contrast_2/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
:random_contrast_2/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipCrandom_contrast_2_stateful_uniform_full_int_rngreadandskip_resource;random_contrast_2/stateful_uniform_full_int/Cast/x:output:06random_contrast_2/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:?
?random_contrast_2/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Arandom_contrast_2/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Arandom_contrast_2/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9random_contrast_2/stateful_uniform_full_int/strided_sliceStridedSliceBrandom_contrast_2/stateful_uniform_full_int/RngReadAndSkip:value:0Hrandom_contrast_2/stateful_uniform_full_int/strided_slice/stack:output:0Jrandom_contrast_2/stateful_uniform_full_int/strided_slice/stack_1:output:0Jrandom_contrast_2/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
3random_contrast_2/stateful_uniform_full_int/BitcastBitcastBrandom_contrast_2/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
Arandom_contrast_2/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Crandom_contrast_2/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Crandom_contrast_2/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;random_contrast_2/stateful_uniform_full_int/strided_slice_1StridedSliceBrandom_contrast_2/stateful_uniform_full_int/RngReadAndSkip:value:0Jrandom_contrast_2/stateful_uniform_full_int/strided_slice_1/stack:output:0Lrandom_contrast_2/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Lrandom_contrast_2/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
5random_contrast_2/stateful_uniform_full_int/Bitcast_1BitcastDrandom_contrast_2/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0q
/random_contrast_2/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
+random_contrast_2/stateful_uniform_full_intStatelessRandomUniformFullIntV2:random_contrast_2/stateful_uniform_full_int/shape:output:0>random_contrast_2/stateful_uniform_full_int/Bitcast_1:output:0<random_contrast_2/stateful_uniform_full_int/Bitcast:output:08random_contrast_2/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	f
random_contrast_2/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ?
random_contrast_2/stackPack4random_contrast_2/stateful_uniform_full_int:output:0%random_contrast_2/zeros_like:output:0*
N*
T0	*
_output_shapes

:v
%random_contrast_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'random_contrast_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'random_contrast_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
random_contrast_2/strided_sliceStridedSlice random_contrast_2/stack:output:0.random_contrast_2/strided_slice/stack:output:00random_contrast_2/strided_slice/stack_1:output:00random_contrast_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_masks
0random_contrast_2/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB s
.random_contrast_2/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *   ?s
.random_contrast_2/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Grandom_contrast_2/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter(random_contrast_2/strided_slice:output:0* 
_output_shapes
::?
Grandom_contrast_2/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
Crandom_contrast_2/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV29random_contrast_2/stateless_random_uniform/shape:output:0Mrandom_contrast_2/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Qrandom_contrast_2/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Prandom_contrast_2/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: ?
.random_contrast_2/stateless_random_uniform/subSub7random_contrast_2/stateless_random_uniform/max:output:07random_contrast_2/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
.random_contrast_2/stateless_random_uniform/mulMulLrandom_contrast_2/stateless_random_uniform/StatelessRandomUniformV2:output:02random_contrast_2/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ?
*random_contrast_2/stateless_random_uniformAddV22random_contrast_2/stateless_random_uniform/mul:z:07random_contrast_2/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
!random_contrast_2/adjust_contrastAdjustContrastv2$random_crop_2/cond/Identity:output:0.random_contrast_2/stateless_random_uniform:z:0*1
_output_shapes
:????????????
*random_contrast_2/adjust_contrast/IdentityIdentity*random_contrast_2/adjust_contrast:output:0*
T0*1
_output_shapes
:????????????
IdentityIdentity3random_contrast_2/adjust_contrast/Identity:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp^center_crop_3/cond;^random_contrast_2/stateful_uniform_full_int/RngReadAndSkip^random_crop_2/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 2(
center_crop_3/condcenter_crop_3/cond2x
:random_contrast_2/stateful_uniform_full_int/RngReadAndSkip:random_contrast_2/stateful_uniform_full_int/RngReadAndSkip2(
random_crop_2/condrandom_crop_2/cond:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block2_conv2_layer_call_and_return_conditional_losses_17121

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????pp?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????pp?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????pp?
 
_user_specified_nameinputs
?
?
G__inference_block4_conv1_layer_call_and_return_conditional_losses_17221

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_block5_conv2_layer_call_fn_17300

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_13236x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_block4_conv2_layer_call_fn_17230

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_13184x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?/
?
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_12719

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity??(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maska
stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *   ?a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: ?
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ?
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: |
adjust_contrastAdjustContrastv2inputsstateless_random_uniform:z:0*1
_output_shapes
:???????????z
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*1
_output_shapes
:???????????z
IdentityIdentity!adjust_contrast/Identity:output:0^NoOp*
T0*1
_output_shapes
:???????????q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
1__inference_random_contrast_2_layer_call_fn_16986

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_12719y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_random_crop_2_layer_call_and_return_conditional_losses_16974

inputs
cond_input_0:	
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
sub/yConst*
_output_shapes
: *
dtype0*
value
B :?S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :?Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: O
Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z
Rank/packedPackRank/packed/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:M
All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z{
	All/inputPackAll/input/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_16838*0
output_shapes
:???????????*"
then_branchR
cond_true_16837d
cond/IdentityIdentitycond:output:0*
T0*1
_output_shapes
:???????????o
IdentityIdentitycond/Identity:output:0^NoOp*
T0*1
_output_shapes
:???????????M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2
condcond:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?2
?
center_crop_3_cond_false_15834"
center_crop_3_cond_placeholder$
 center_crop_3_cond_placeholder_1#
center_crop_3_cond_shape_inputs
center_crop_3_cond_identityg
center_crop_3/cond/ShapeShapecenter_crop_3_cond_shape_inputs*
T0*
_output_shapes
:y
&center_crop_3/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
(center_crop_3/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(center_crop_3/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 center_crop_3/cond/strided_sliceStridedSlice!center_crop_3/cond/Shape:output:0/center_crop_3/cond/strided_slice/stack:output:01center_crop_3/cond/strided_slice/stack_1:output:01center_crop_3/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
(center_crop_3/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*center_crop_3/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*center_crop_3/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"center_crop_3/cond/strided_slice_1StridedSlice!center_crop_3/cond/Shape:output:01center_crop_3/cond/strided_slice_1/stack:output:03center_crop_3/cond/strided_slice_1/stack_1:output:03center_crop_3/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
center_crop_3/cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
center_crop_3/cond/mulMul+center_crop_3/cond/strided_slice_1:output:0!center_crop_3/cond/mul/y:output:0*
T0*
_output_shapes
: k
center_crop_3/cond/CastCastcenter_crop_3/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: a
center_crop_3/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
center_crop_3/cond/truedivRealDivcenter_crop_3/cond/Cast:y:0%center_crop_3/cond/truediv/y:output:0*
T0*
_output_shapes
: q
center_crop_3/cond/Cast_1Castcenter_crop_3/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: ]
center_crop_3/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
center_crop_3/cond/mul_1Mul)center_crop_3/cond/strided_slice:output:0#center_crop_3/cond/mul_1/y:output:0*
T0*
_output_shapes
: o
center_crop_3/cond/Cast_2Castcenter_crop_3/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
center_crop_3/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
center_crop_3/cond/truediv_1RealDivcenter_crop_3/cond/Cast_2:y:0'center_crop_3/cond/truediv_1/y:output:0*
T0*
_output_shapes
: s
center_crop_3/cond/Cast_3Cast center_crop_3/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
center_crop_3/cond/MinimumMinimum)center_crop_3/cond/strided_slice:output:0center_crop_3/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
center_crop_3/cond/Minimum_1Minimum+center_crop_3/cond/strided_slice_1:output:0center_crop_3/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
center_crop_3/cond/subSub)center_crop_3/cond/strided_slice:output:0center_crop_3/cond/Minimum:z:0*
T0*
_output_shapes
: m
center_crop_3/cond/Cast_4Castcenter_crop_3/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: c
center_crop_3/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
center_crop_3/cond/truediv_2RealDivcenter_crop_3/cond/Cast_4:y:0'center_crop_3/cond/truediv_2/y:output:0*
T0*
_output_shapes
: s
center_crop_3/cond/Cast_5Cast center_crop_3/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
center_crop_3/cond/sub_1Sub+center_crop_3/cond/strided_slice_1:output:0 center_crop_3/cond/Minimum_1:z:0*
T0*
_output_shapes
: o
center_crop_3/cond/Cast_6Castcenter_crop_3/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
center_crop_3/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
center_crop_3/cond/truediv_3RealDivcenter_crop_3/cond/Cast_6:y:0'center_crop_3/cond/truediv_3/y:output:0*
T0*
_output_shapes
: s
center_crop_3/cond/Cast_7Cast center_crop_3/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: \
center_crop_3/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : \
center_crop_3/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
center_crop_3/cond/stackPack#center_crop_3/cond/stack/0:output:0center_crop_3/cond/Cast_5:y:0center_crop_3/cond/Cast_7:y:0#center_crop_3/cond/stack/3:output:0*
N*
T0*
_output_shapes
:g
center_crop_3/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????g
center_crop_3/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
center_crop_3/cond/stack_1Pack%center_crop_3/cond/stack_1/0:output:0center_crop_3/cond/Minimum:z:0 center_crop_3/cond/Minimum_1:z:0%center_crop_3/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
center_crop_3/cond/SliceSlicecenter_crop_3_cond_shape_inputs!center_crop_3/cond/stack:output:0#center_crop_3/cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????o
center_crop_3/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
(center_crop_3/cond/resize/ResizeBilinearResizeBilinear!center_crop_3/cond/Slice:output:0'center_crop_3/cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
center_crop_3/cond/IdentityIdentity9center_crop_3/cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"C
center_crop_3_cond_identity$center_crop_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_17071

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?'
j
cond_false_16519
cond_placeholder
cond_placeholder_1
cond_shape_inputs
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM

cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: O
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :?f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Cf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
G
+__inference_block5_pool_layer_call_fn_17336

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_13024?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_block3_pool_layer_call_fn_17196

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_13000?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_block5_conv1_layer_call_and_return_conditional_losses_17291

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15806

inputs(
random_crop_2_cond_input_0:	
identity??center_crop_3/cond?random_crop_2/condI
center_crop_3/ShapeShapeinputs*
T0*
_output_shapes
:t
!center_crop_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
#center_crop_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#center_crop_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
center_crop_3/strided_sliceStridedSlicecenter_crop_3/Shape:output:0*center_crop_3/strided_slice/stack:output:0,center_crop_3/strided_slice/stack_1:output:0,center_crop_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
center_crop_3/sub/yConst*
_output_shapes
: *
dtype0*
value
B :?}
center_crop_3/subSub$center_crop_3/strided_slice:output:0center_crop_3/sub/y:output:0*
T0*
_output_shapes
: v
#center_crop_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????x
%center_crop_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%center_crop_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
center_crop_3/strided_slice_1StridedSlicecenter_crop_3/Shape:output:0,center_crop_3/strided_slice_1/stack:output:0.center_crop_3/strided_slice_1/stack_1:output:0.center_crop_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
center_crop_3/sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
center_crop_3/sub_1Sub&center_crop_3/strided_slice_1:output:0center_crop_3/sub_1/y:output:0*
T0*
_output_shapes
: ^
center_crop_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
center_crop_3/GreaterEqualGreaterEqualcenter_crop_3/sub:z:0%center_crop_3/GreaterEqual/y:output:0*
T0*
_output_shapes
: `
center_crop_3/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
center_crop_3/GreaterEqual_1GreaterEqualcenter_crop_3/sub_1:z:0'center_crop_3/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
center_crop_3/Rank/packedPackcenter_crop_3/GreaterEqual:z:0 center_crop_3/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:T
center_crop_3/RankConst*
_output_shapes
: *
dtype0*
value	B :[
center_crop_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : [
center_crop_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
center_crop_3/rangeRange"center_crop_3/range/start:output:0center_crop_3/Rank:output:0"center_crop_3/range/delta:output:0*
_output_shapes
:?
center_crop_3/All/inputPackcenter_crop_3/GreaterEqual:z:0 center_crop_3/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:p
center_crop_3/AllAll center_crop_3/All/input:output:0center_crop_3/range:output:0*
_output_shapes
: ?
center_crop_3/condIfcenter_crop_3/All:output:0center_crop_3/sub:z:0center_crop_3/sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *1
else_branch"R 
center_crop_3_cond_false_15535*0
output_shapes
:???????????*0
then_branch!R
center_crop_3_cond_true_15534?
center_crop_3/cond/IdentityIdentitycenter_crop_3/cond:output:0*
T0*1
_output_shapes
:???????????g
random_crop_2/ShapeShape$center_crop_3/cond/Identity:output:0*
T0*
_output_shapes
:t
!random_crop_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
#random_crop_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#random_crop_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop_2/strided_sliceStridedSlicerandom_crop_2/Shape:output:0*random_crop_2/strided_slice/stack:output:0,random_crop_2/strided_slice/stack_1:output:0,random_crop_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
random_crop_2/sub/yConst*
_output_shapes
: *
dtype0*
value
B :?}
random_crop_2/subSub$random_crop_2/strided_slice:output:0random_crop_2/sub/y:output:0*
T0*
_output_shapes
: v
#random_crop_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????x
%random_crop_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%random_crop_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
random_crop_2/strided_slice_1StridedSlicerandom_crop_2/Shape:output:0,random_crop_2/strided_slice_1/stack:output:0.random_crop_2/strided_slice_1/stack_1:output:0.random_crop_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
random_crop_2/sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
random_crop_2/sub_1Sub&random_crop_2/strided_slice_1:output:0random_crop_2/sub_1/y:output:0*
T0*
_output_shapes
: ^
random_crop_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop_2/GreaterEqualGreaterEqualrandom_crop_2/sub:z:0%random_crop_2/GreaterEqual/y:output:0*
T0*
_output_shapes
: `
random_crop_2/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
random_crop_2/GreaterEqual_1GreaterEqualrandom_crop_2/sub_1:z:0'random_crop_2/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ]
random_crop_2/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
random_crop_2/Rank/packedPack$random_crop_2/Rank/packed/0:output:0random_crop_2/GreaterEqual:z:0 random_crop_2/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:T
random_crop_2/RankConst*
_output_shapes
: *
dtype0*
value	B :[
random_crop_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : [
random_crop_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
random_crop_2/rangeRange"random_crop_2/range/start:output:0random_crop_2/Rank:output:0"random_crop_2/range/delta:output:0*
_output_shapes
:[
random_crop_2/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
random_crop_2/All/inputPack"random_crop_2/All/input/0:output:0random_crop_2/GreaterEqual:z:0 random_crop_2/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:p
random_crop_2/AllAll random_crop_2/All/input:output:0random_crop_2/range:output:0*
_output_shapes
: ?
random_crop_2/condIfrandom_crop_2/All:output:0random_crop_2_cond_input_0random_crop_2/sub:z:0random_crop_2/sub_1:z:0$center_crop_3/cond/Identity:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *1
else_branch"R 
random_crop_2_cond_false_15670*0
output_shapes
:???????????*0
then_branch!R
random_crop_2_cond_true_15669?
random_crop_2/cond/IdentityIdentityrandom_crop_2/cond:output:0*
T0*1
_output_shapes
:???????????}
IdentityIdentity$random_crop_2/cond/Identity:output:0^NoOp*
T0*1
_output_shapes
:???????????p
NoOpNoOp^center_crop_3/cond^random_crop_2/cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2(
center_crop_3/condcenter_crop_3/cond2(
random_crop_2/condrandom_crop_2/cond:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?'
?
cond_false_12518
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_shape_inputs
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM

cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: O
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :?f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Cf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?3
?
random_crop_2_cond_false_15969"
random_crop_2_cond_placeholder$
 random_crop_2_cond_placeholder_1$
 random_crop_2_cond_placeholder_28
4random_crop_2_cond_shape_center_crop_3_cond_identity
random_crop_2_cond_identity|
random_crop_2/cond/ShapeShape4random_crop_2_cond_shape_center_crop_3_cond_identity*
T0*
_output_shapes
:y
&random_crop_2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
(random_crop_2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????r
(random_crop_2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 random_crop_2/cond/strided_sliceStridedSlice!random_crop_2/cond/Shape:output:0/random_crop_2/cond/strided_slice/stack:output:01random_crop_2/cond/strided_slice/stack_1:output:01random_crop_2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
(random_crop_2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*random_crop_2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*random_crop_2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"random_crop_2/cond/strided_slice_1StridedSlice!random_crop_2/cond/Shape:output:01random_crop_2/cond/strided_slice_1/stack:output:03random_crop_2/cond/strided_slice_1/stack_1:output:03random_crop_2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
random_crop_2/cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
random_crop_2/cond/mulMul+random_crop_2/cond/strided_slice_1:output:0!random_crop_2/cond/mul/y:output:0*
T0*
_output_shapes
: k
random_crop_2/cond/CastCastrandom_crop_2/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: a
random_crop_2/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
random_crop_2/cond/truedivRealDivrandom_crop_2/cond/Cast:y:0%random_crop_2/cond/truediv/y:output:0*
T0*
_output_shapes
: q
random_crop_2/cond/Cast_1Castrandom_crop_2/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: ]
random_crop_2/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
random_crop_2/cond/mul_1Mul)random_crop_2/cond/strided_slice:output:0#random_crop_2/cond/mul_1/y:output:0*
T0*
_output_shapes
: o
random_crop_2/cond/Cast_2Castrandom_crop_2/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_2/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
random_crop_2/cond/truediv_1RealDivrandom_crop_2/cond/Cast_2:y:0'random_crop_2/cond/truediv_1/y:output:0*
T0*
_output_shapes
: s
random_crop_2/cond/Cast_3Cast random_crop_2/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
random_crop_2/cond/MinimumMinimum)random_crop_2/cond/strided_slice:output:0random_crop_2/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
random_crop_2/cond/Minimum_1Minimum+random_crop_2/cond/strided_slice_1:output:0random_crop_2/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
random_crop_2/cond/subSub)random_crop_2/cond/strided_slice:output:0random_crop_2/cond/Minimum:z:0*
T0*
_output_shapes
: m
random_crop_2/cond/Cast_4Castrandom_crop_2/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_2/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
random_crop_2/cond/truediv_2RealDivrandom_crop_2/cond/Cast_4:y:0'random_crop_2/cond/truediv_2/y:output:0*
T0*
_output_shapes
: s
random_crop_2/cond/Cast_5Cast random_crop_2/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
random_crop_2/cond/sub_1Sub+random_crop_2/cond/strided_slice_1:output:0 random_crop_2/cond/Minimum_1:z:0*
T0*
_output_shapes
: o
random_crop_2/cond/Cast_6Castrandom_crop_2/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: c
random_crop_2/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
random_crop_2/cond/truediv_3RealDivrandom_crop_2/cond/Cast_6:y:0'random_crop_2/cond/truediv_3/y:output:0*
T0*
_output_shapes
: s
random_crop_2/cond/Cast_7Cast random_crop_2/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: \
random_crop_2/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : \
random_crop_2/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
random_crop_2/cond/stackPack#random_crop_2/cond/stack/0:output:0random_crop_2/cond/Cast_5:y:0random_crop_2/cond/Cast_7:y:0#random_crop_2/cond/stack/3:output:0*
N*
T0*
_output_shapes
:g
random_crop_2/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????g
random_crop_2/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
random_crop_2/cond/stack_1Pack%random_crop_2/cond/stack_1/0:output:0random_crop_2/cond/Minimum:z:0 random_crop_2/cond/Minimum_1:z:0%random_crop_2/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
random_crop_2/cond/SliceSlice4random_crop_2_cond_shape_center_crop_3_cond_identity!random_crop_2/cond/stack:output:0#random_crop_2/cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????o
random_crop_2/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
(random_crop_2/cond/resize/ResizeBilinearResizeBilinear!random_crop_2/cond/Slice:output:0'random_crop_2/cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
random_crop_2/cond/IdentityIdentity9random_crop_2/cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"C
random_crop_2_cond_identity$random_crop_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
G__inference_block4_conv2_layer_call_and_return_conditional_losses_17241

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_14502

inputs
unknown
	unknown_0:	#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?

unknown_27:	?

unknown_28:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_13939o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?
G
+__inference_block2_pool_layer_call_fn_17126

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_12988?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_random_crop_2_layer_call_and_return_conditional_losses_12654

inputs
cond_input_0:	
identity??cond;
ShapeShapeinputs*
T0*
_output_shapes
:f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
sub/yConst*
_output_shapes
: *
dtype0*
value
B :?S
subSubstrided_slice:output:0sub/y:output:0*
T0*
_output_shapes
: h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :?Y
sub_1Substrided_slice_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: P
GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterEqualGreaterEqualsub:z:0GreaterEqual/y:output:0*
T0*
_output_shapes
: R
GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : e
GreaterEqual_1GreaterEqual	sub_1:z:0GreaterEqual_1/y:output:0*
T0*
_output_shapes
: O
Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z 
Rank/packedPackRank/packed/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:M
All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z {
	All/inputPackAll/input/0:output:0GreaterEqual:z:0GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:F
AllAllAll/input:output:0range:output:0*
_output_shapes
: ?
condIfAll:output:0cond_input_0sub:z:0	sub_1:z:0inputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *#
else_branchR
cond_false_12518*0
output_shapes
:???????????*"
then_branchR
cond_true_12517d
cond/IdentityIdentitycond:output:0*
T0*1
_output_shapes
:???????????o
IdentityIdentitycond/Identity:output:0^NoOp*
T0*1
_output_shapes
:???????????M
NoOpNoOp^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2
condcond:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?/
?
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_17031

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity??(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maska
stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *   ?a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: ?
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ?
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: |
adjust_contrastAdjustContrastv2inputsstateless_random_uniform:z:0*1
_output_shapes
:???????????z
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*1
_output_shapes
:???????????z
IdentityIdentity!adjust_contrast/Identity:output:0^NoOp*
T0*1
_output_shapes
:???????????q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?g
?
cond_true_16837;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/Assert?$cond/stateful_uniform/RngReadAndSkipe
cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:[
cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : _
cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????e
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
cond/stateful_uniform/ProdProd$cond/stateful_uniform/shape:output:0$cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ^
cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :y
cond/stateful_uniform/Cast_1Cast#cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$cond/stateful_uniform/RngReadAndSkipRngReadAndSkip-cond_stateful_uniform_rngreadandskip_resource%cond/stateful_uniform/Cast/x:output:0 cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:s
)cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
cond/stateful_uniform/BitcastBitcast,cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0u
+cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
cond/stateful_uniformStatelessRandomUniformIntV2$cond/stateful_uniform/shape:output:0(cond/stateful_uniform/Bitcast_1:output:0&cond/stateful_uniform/Bitcast:output:0"cond/stateful_uniform/alg:output:0"cond/stateful_uniform/min:output:0"cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0b
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/stateful_uniform:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :U
cond/addAddV2cond_add_subcond/add/y:output:0*
T0*
_output_shapes
: `
cond/modFloorModcond/strided_slice:output:0cond/add:z:0*
T0*
_output_shapes
: d
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :]

cond/add_1AddV2cond_add_1_sub_1cond/add_1/y:output:0*
T0*
_output_shapes
: f

cond/mod_1FloorModcond/strided_slice_1:output:0cond/add_1:z:0*
T0*
_output_shapes
: u
cond/crop_to_bounding_box/ShapeShape&cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 b
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: h
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 d
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: j
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*1
_output_shapes
:???????????c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :?f
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*1
_output_shapes
:????????????
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
'__inference_model_3_layer_call_fn_14569

inputs
unknown
	unknown_0:	
	unknown_1:	#
	unknown_2:@
	unknown_3:@#
	unknown_4:@@
	unknown_5:@$
	unknown_6:@?
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?&

unknown_10:??

unknown_11:	?&

unknown_12:??

unknown_13:	?&

unknown_14:??

unknown_15:	?&

unknown_16:??

unknown_17:	?&

unknown_18:??

unknown_19:	?&

unknown_20:??

unknown_21:	?&

unknown_22:??

unknown_23:	?&

unknown_24:??

unknown_25:	?&

unknown_26:??

unknown_27:	?

unknown_28:	?

unknown_29:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_14153o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_17271

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ҷ
?
!__inference__traced_restore_17667
file_prefix2
assignvariableop_dense_3_kernel:	?-
assignvariableop_1_dense_3_bias:)
assignvariableop_2_rmsprop_iter:	 *
 assignvariableop_3_rmsprop_decay: 2
(assignvariableop_4_rmsprop_learning_rate: -
#assignvariableop_5_rmsprop_momentum: (
assignvariableop_6_rmsprop_rho: @
&assignvariableop_7_block1_conv1_kernel:@2
$assignvariableop_8_block1_conv1_bias:@@
&assignvariableop_9_block1_conv2_kernel:@@3
%assignvariableop_10_block1_conv2_bias:@B
'assignvariableop_11_block2_conv1_kernel:@?4
%assignvariableop_12_block2_conv1_bias:	?C
'assignvariableop_13_block2_conv2_kernel:??4
%assignvariableop_14_block2_conv2_bias:	?C
'assignvariableop_15_block3_conv1_kernel:??4
%assignvariableop_16_block3_conv1_bias:	?C
'assignvariableop_17_block3_conv2_kernel:??4
%assignvariableop_18_block3_conv2_bias:	?C
'assignvariableop_19_block3_conv3_kernel:??4
%assignvariableop_20_block3_conv3_bias:	?C
'assignvariableop_21_block4_conv1_kernel:??4
%assignvariableop_22_block4_conv1_bias:	?C
'assignvariableop_23_block4_conv2_kernel:??4
%assignvariableop_24_block4_conv2_bias:	?C
'assignvariableop_25_block4_conv3_kernel:??4
%assignvariableop_26_block4_conv3_bias:	?C
'assignvariableop_27_block5_conv1_kernel:??4
%assignvariableop_28_block5_conv1_bias:	?C
'assignvariableop_29_block5_conv2_kernel:??4
%assignvariableop_30_block5_conv2_bias:	?C
'assignvariableop_31_block5_conv3_kernel:??4
%assignvariableop_32_block5_conv3_bias:	?#
assignvariableop_33_total: #
assignvariableop_34_count: %
assignvariableop_35_total_1: %
assignvariableop_36_count_1: =
/assignvariableop_37_random_crop_2_cond_statevar:	<
.assignvariableop_38_random_contrast_2_statevar:	A
.assignvariableop_39_rmsprop_dense_3_kernel_rms:	?:
,assignvariableop_40_rmsprop_dense_3_bias_rms:O
3assignvariableop_41_rmsprop_block5_conv1_kernel_rms:??@
1assignvariableop_42_rmsprop_block5_conv1_bias_rms:	?O
3assignvariableop_43_rmsprop_block5_conv2_kernel_rms:??@
1assignvariableop_44_rmsprop_block5_conv2_bias_rms:	?O
3assignvariableop_45_rmsprop_block5_conv3_kernel_rms:??@
1assignvariableop_46_rmsprop_block5_conv3_bias_rms:	?
identity_48??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*?
value?B?0B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-3/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-3/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/20/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/21/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/22/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/23/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/24/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBEvariables/25/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220			[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_rmsprop_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_rmsprop_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_rmsprop_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_rmsprop_momentumIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_rhoIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp&assignvariableop_7_block1_conv1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_block1_conv1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp&assignvariableop_9_block1_conv2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp%assignvariableop_10_block1_conv2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_block2_conv1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_block2_conv1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_block2_conv2_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_block2_conv2_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_block3_conv1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_block3_conv1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_block3_conv2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_block3_conv2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_block3_conv3_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp%assignvariableop_20_block3_conv3_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_block4_conv1_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_block4_conv1_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_block4_conv2_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_block4_conv2_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_block4_conv3_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_block4_conv3_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_block5_conv1_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_block5_conv1_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_block5_conv2_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_block5_conv2_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_block5_conv3_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_block5_conv3_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp/assignvariableop_37_random_crop_2_cond_statevarIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp.assignvariableop_38_random_contrast_2_statevarIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp.assignvariableop_39_rmsprop_dense_3_kernel_rmsIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp,assignvariableop_40_rmsprop_dense_3_bias_rmsIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp3assignvariableop_41_rmsprop_block5_conv1_kernel_rmsIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp1assignvariableop_42_rmsprop_block5_conv1_bias_rmsIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp3assignvariableop_43_rmsprop_block5_conv2_kernel_rmsIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp1assignvariableop_44_rmsprop_block5_conv2_bias_rmsIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp3assignvariableop_45_rmsprop_block5_conv3_kernel_rmsIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp1assignvariableop_46_rmsprop_block5_conv3_bias_rmsIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_48IdentityIdentity_47:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_48Identity_48:output:0*s
_input_shapesb
`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462(
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
?
?
G__inference_block3_conv3_layer_call_and_return_conditional_losses_17191

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????88?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
G__inference_block5_conv1_layer_call_and_return_conditional_losses_13219

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_block1_conv1_layer_call_fn_17040

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_13045y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_block4_conv1_layer_call_fn_17210

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_13167x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_block5_conv3_layer_call_fn_17320

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_13253x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block4_conv1_layer_call_and_return_conditional_losses_13167

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block5_conv3_layer_call_and_return_conditional_losses_13253

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12931

inputs!
random_crop_2_12924:	%
random_contrast_2_12927:	
identity??%center_crop_3/StatefulPartitionedCall?)random_contrast_2/StatefulPartitionedCall?%random_crop_2/StatefulPartitionedCall?
%center_crop_3/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_center_crop_3_layer_call_and_return_conditional_losses_12486?
%random_crop_2/StatefulPartitionedCallStatefulPartitionedCall.center_crop_3/StatefulPartitionedCall:output:0random_crop_2_12924*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_random_crop_2_layer_call_and_return_conditional_losses_12900?
)random_contrast_2/StatefulPartitionedCallStatefulPartitionedCall.random_crop_2/StatefulPartitionedCall:output:0random_contrast_2_12927*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_12719?
IdentityIdentity2random_contrast_2/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp&^center_crop_3/StatefulPartitionedCall*^random_contrast_2/StatefulPartitionedCall&^random_crop_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 2N
%center_crop_3/StatefulPartitionedCall%center_crop_3/StatefulPartitionedCall2V
)random_contrast_2/StatefulPartitionedCall)random_contrast_2/StatefulPartitionedCall2N
%random_crop_2/StatefulPartitionedCall%random_crop_2/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
@__inference_vgg16_layer_call_and_return_conditional_losses_16356

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@?;
,block2_conv1_biasadd_readvariableop_resource:	?G
+block2_conv2_conv2d_readvariableop_resource:??;
,block2_conv2_biasadd_readvariableop_resource:	?G
+block3_conv1_conv2d_readvariableop_resource:??;
,block3_conv1_biasadd_readvariableop_resource:	?G
+block3_conv2_conv2d_readvariableop_resource:??;
,block3_conv2_biasadd_readvariableop_resource:	?G
+block3_conv3_conv2d_readvariableop_resource:??;
,block3_conv3_biasadd_readvariableop_resource:	?G
+block4_conv1_conv2d_readvariableop_resource:??;
,block4_conv1_biasadd_readvariableop_resource:	?G
+block4_conv2_conv2d_readvariableop_resource:??;
,block4_conv2_biasadd_readvariableop_resource:	?G
+block4_conv3_conv2d_readvariableop_resource:??;
,block4_conv3_biasadd_readvariableop_resource:	?G
+block5_conv1_conv2d_readvariableop_resource:??;
,block5_conv1_biasadd_readvariableop_resource:	?G
+block5_conv2_conv2d_readvariableop_resource:??;
,block5_conv2_biasadd_readvariableop_resource:	?G
+block5_conv3_conv2d_readvariableop_resource:??;
,block5_conv3_biasadd_readvariableop_resource:	?
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?#block5_conv2/BiasAdd/ReadVariableOp?"block5_conv2/Conv2D/ReadVariableOp?#block5_conv3/BiasAdd/ReadVariableOp?"block5_conv3/Conv2D/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?s
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp??
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?s
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp??
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?s
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?s
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?s
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????s
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_13024

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
-__inference_center_crop_3_layer_call_fn_16491

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_center_crop_3_layer_call_and_return_conditional_losses_12486y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_12988

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_17131

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?h
?
*sequential_2_center_crop_3_cond_true_14602O
Ksequential_2_center_crop_3_cond_truediv_cast_sequential_2_center_crop_3_subS
Osequential_2_center_crop_3_cond_truediv_1_cast_sequential_2_center_crop_3_sub_1W
Ssequential_2_center_crop_3_cond_crop_to_bounding_box_shape_tf_nn_bias_add_3_biasadd,
(sequential_2_center_crop_3_cond_identity??Bsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert?Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert?Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert?Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assertk
)sequential_2/center_crop_3/cond/truediv/yConst*
_output_shapes
: *
dtype0*
value	B :?
,sequential_2/center_crop_3/cond/truediv/CastCastKsequential_2_center_crop_3_cond_truediv_cast_sequential_2_center_crop_3_sub*

DstT0*

SrcT0*
_output_shapes
: ?
.sequential_2/center_crop_3/cond/truediv/Cast_1Cast2sequential_2/center_crop_3/cond/truediv/y:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
'sequential_2/center_crop_3/cond/truedivRealDiv0sequential_2/center_crop_3/cond/truediv/Cast:y:02sequential_2/center_crop_3/cond/truediv/Cast_1:y:0*
T0*
_output_shapes
: ?
$sequential_2/center_crop_3/cond/CastCast+sequential_2/center_crop_3/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: m
+sequential_2/center_crop_3/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
.sequential_2/center_crop_3/cond/truediv_1/CastCastOsequential_2_center_crop_3_cond_truediv_1_cast_sequential_2_center_crop_3_sub_1*

DstT0*

SrcT0*
_output_shapes
: ?
0sequential_2/center_crop_3/cond/truediv_1/Cast_1Cast4sequential_2/center_crop_3/cond/truediv_1/y:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
)sequential_2/center_crop_3/cond/truediv_1RealDiv2sequential_2/center_crop_3/cond/truediv_1/Cast:y:04sequential_2/center_crop_3/cond/truediv_1/Cast_1:y:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_1Cast-sequential_2/center_crop_3/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
:sequential_2/center_crop_3/cond/crop_to_bounding_box/ShapeShapeSsequential_2_center_crop_3_cond_crop_to_bounding_box_shape_tf_nn_bias_add_3_biasadd*
T0*
_output_shapes
:?
<sequential_2/center_crop_3/cond/crop_to_bounding_box/unstackUnpackCsequential_2/center_crop_3/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num?
Csequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Asequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqualGreaterEqual*sequential_2/center_crop_3/cond/Cast_1:y:0Lsequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Asequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Isequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Bsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertAssertEsequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual:z:0Rsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
Esequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Csequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual(sequential_2/center_crop_3/cond/Cast:y:0Nsequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Ksequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertAssertGsequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1:z:0Tsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0C^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 }
:sequential_2/center_crop_3/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
8sequential_2/center_crop_3/cond/crop_to_bounding_box/addAddV2Csequential_2/center_crop_3/cond/crop_to_bounding_box/add/x:output:0*sequential_2/center_crop_3/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
@sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
>sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual	LessEqual<sequential_2/center_crop_3/cond/crop_to_bounding_box/add:z:0Isequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Ksequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertAssertBsequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual:z:0Tsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0E^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 
<sequential_2/center_crop_3/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
:sequential_2/center_crop_3/cond/crop_to_bounding_box/add_1AddV2Esequential_2/center_crop_3/cond/crop_to_bounding_box/add_1/x:output:0(sequential_2/center_crop_3/cond/Cast:y:0*
T0*
_output_shapes
: ?
Bsequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
@sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1	LessEqual>sequential_2/center_crop_3/cond/crop_to_bounding_box/add_1:z:0Ksequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Ksequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/AssertAssertDsequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1:z:0Tsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0E^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
Gsequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependencyIdentitySsequential_2_center_crop_3_cond_crop_to_bounding_box_shape_tf_nn_bias_add_3_biasaddC^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert*
T0*+
_class!
loc:@tf.nn.bias_add_3/BiasAdd*1
_output_shapes
:???????????~
<sequential_2/center_crop_3/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : ~
<sequential_2/center_crop_3/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
:sequential_2/center_crop_3/cond/crop_to_bounding_box/stackPackEsequential_2/center_crop_3/cond/crop_to_bounding_box/stack/0:output:0(sequential_2/center_crop_3/cond/Cast:y:0*sequential_2/center_crop_3/cond/Cast_1:y:0Esequential_2/center_crop_3/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
<sequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_1ShapePsequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Hsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Jsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Jsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_sliceStridedSliceEsequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_1:output:0Qsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack:output:0Ssequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Ssequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<sequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_2ShapePsequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Jsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Lsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Lsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1StridedSliceEsequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_2:output:0Ssequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Usequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Usequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :??
>sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
<sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1PackKsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice:output:0Gsequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/1:output:0Gsequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/2:output:0Msequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
:sequential_2/center_crop_3/cond/crop_to_bounding_box/SliceSlicePsequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependency:output:0Csequential_2/center_crop_3/cond/crop_to_bounding_box/stack:output:0Esequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
(sequential_2/center_crop_3/cond/IdentityIdentityCsequential_2/center_crop_3/cond/crop_to_bounding_box/Slice:output:0%^sequential_2/center_crop_3/cond/NoOp*
T0*1
_output_shapes
:????????????
$sequential_2/center_crop_3/cond/NoOpNoOpC^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_2_center_crop_3_cond_identity1sequential_2/center_crop_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????2?
Bsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertBsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert2?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertDsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert2?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertDsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert2?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/AssertDsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?q
?
2model_3_sequential_2_center_crop_3_cond_true_11966_
[model_3_sequential_2_center_crop_3_cond_truediv_cast_model_3_sequential_2_center_crop_3_subc
_model_3_sequential_2_center_crop_3_cond_truediv_1_cast_model_3_sequential_2_center_crop_3_sub_1g
cmodel_3_sequential_2_center_crop_3_cond_crop_to_bounding_box_shape_model_3_tf_nn_bias_add_3_biasadd4
0model_3_sequential_2_center_crop_3_cond_identity??Jmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert?Lmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert?Lmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert?Lmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Asserts
1model_3/sequential_2/center_crop_3/cond/truediv/yConst*
_output_shapes
: *
dtype0*
value	B :?
4model_3/sequential_2/center_crop_3/cond/truediv/CastCast[model_3_sequential_2_center_crop_3_cond_truediv_cast_model_3_sequential_2_center_crop_3_sub*

DstT0*

SrcT0*
_output_shapes
: ?
6model_3/sequential_2/center_crop_3/cond/truediv/Cast_1Cast:model_3/sequential_2/center_crop_3/cond/truediv/y:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
/model_3/sequential_2/center_crop_3/cond/truedivRealDiv8model_3/sequential_2/center_crop_3/cond/truediv/Cast:y:0:model_3/sequential_2/center_crop_3/cond/truediv/Cast_1:y:0*
T0*
_output_shapes
: ?
,model_3/sequential_2/center_crop_3/cond/CastCast3model_3/sequential_2/center_crop_3/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: u
3model_3/sequential_2/center_crop_3/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
6model_3/sequential_2/center_crop_3/cond/truediv_1/CastCast_model_3_sequential_2_center_crop_3_cond_truediv_1_cast_model_3_sequential_2_center_crop_3_sub_1*

DstT0*

SrcT0*
_output_shapes
: ?
8model_3/sequential_2/center_crop_3/cond/truediv_1/Cast_1Cast<model_3/sequential_2/center_crop_3/cond/truediv_1/y:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
1model_3/sequential_2/center_crop_3/cond/truediv_1RealDiv:model_3/sequential_2/center_crop_3/cond/truediv_1/Cast:y:0<model_3/sequential_2/center_crop_3/cond/truediv_1/Cast_1:y:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/center_crop_3/cond/Cast_1Cast5model_3/sequential_2/center_crop_3/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
Bmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/ShapeShapecmodel_3_sequential_2_center_crop_3_cond_crop_to_bounding_box_shape_model_3_tf_nn_bias_add_3_biasadd*
T0*
_output_shapes
:?
Dmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/unstackUnpackKmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num?
Kmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Imodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqualGreaterEqual2model_3/sequential_2/center_crop_3/cond/Cast_1:y:0Tmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Imodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Qmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Jmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertAssertMmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual:z:0Zmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
Mmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Kmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual0model_3/sequential_2/center_crop_3/cond/Cast:y:0Vmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
Kmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Smodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Lmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertAssertOmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1:z:0\model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0K^model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 ?
Bmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
@model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/addAddV2Kmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/add/x:output:02model_3/sequential_2/center_crop_3/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
Hmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
Fmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual	LessEqualDmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/add:z:0Qmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
Kmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Smodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Lmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertAssertJmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual:z:0\model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0M^model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 ?
Dmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
Bmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/add_1AddV2Mmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/add_1/x:output:00model_3/sequential_2/center_crop_3/cond/Cast:y:0*
T0*
_output_shapes
: ?
Jmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
Hmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1	LessEqualFmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/add_1:z:0Smodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
Kmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Smodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Lmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/AssertAssertLmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1:z:0\model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0M^model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
Omodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependencyIdentitycmodel_3_sequential_2_center_crop_3_cond_crop_to_bounding_box_shape_model_3_tf_nn_bias_add_3_biasaddK^model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertM^model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertM^model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertM^model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert*
T0*3
_class)
'%loc:@model_3/tf.nn.bias_add_3/BiasAdd*1
_output_shapes
:????????????
Dmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
Bmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stackPackMmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stack/0:output:00model_3/sequential_2/center_crop_3/cond/Cast:y:02model_3/sequential_2/center_crop_3/cond/Cast_1:y:0Mmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
Dmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_1ShapeXmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Pmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Rmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Rmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_sliceStridedSliceMmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_1:output:0Ymodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack:output:0[model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_1:output:0[model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Dmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_2ShapeXmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Rmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Tmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Tmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1StridedSliceMmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_2:output:0[model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack:output:0]model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0]model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Fmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :??
Fmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
Dmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1PackSmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice:output:0Omodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/1:output:0Omodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/2:output:0Umodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
Bmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/SliceSliceXmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependency:output:0Kmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stack:output:0Mmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
0model_3/sequential_2/center_crop_3/cond/IdentityIdentityKmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Slice:output:0-^model_3/sequential_2/center_crop_3/cond/NoOp*
T0*1
_output_shapes
:????????????
,model_3/sequential_2/center_crop_3/cond/NoOpNoOpK^model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertM^model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertM^model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertM^model_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert*"
_acd_function_control_output(*
_output_shapes
 "m
0model_3_sequential_2_center_crop_3_cond_identity9model_3/sequential_2/center_crop_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????2?
Jmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertJmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert2?
Lmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertLmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert2?
Lmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertLmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert2?
Lmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/AssertLmodel_3/sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
%__inference_vgg16_layer_call_fn_16199

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_13261x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
h
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_16990

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block2_conv1_layer_call_and_return_conditional_losses_13080

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????pp?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?g
?
cond_true_12763;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/Assert?$cond/stateful_uniform/RngReadAndSkipe
cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:[
cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : _
cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????e
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
cond/stateful_uniform/ProdProd$cond/stateful_uniform/shape:output:0$cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ^
cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :y
cond/stateful_uniform/Cast_1Cast#cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$cond/stateful_uniform/RngReadAndSkipRngReadAndSkip-cond_stateful_uniform_rngreadandskip_resource%cond/stateful_uniform/Cast/x:output:0 cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:s
)cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
cond/stateful_uniform/BitcastBitcast,cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0u
+cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
cond/stateful_uniformStatelessRandomUniformIntV2$cond/stateful_uniform/shape:output:0(cond/stateful_uniform/Bitcast_1:output:0&cond/stateful_uniform/Bitcast:output:0"cond/stateful_uniform/alg:output:0"cond/stateful_uniform/min:output:0"cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0b
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/stateful_uniform:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :U
cond/addAddV2cond_add_subcond/add/y:output:0*
T0*
_output_shapes
: `
cond/modFloorModcond/strided_slice:output:0cond/add:z:0*
T0*
_output_shapes
: d
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :]

cond/add_1AddV2cond_add_1_sub_1cond/add_1/y:output:0*
T0*
_output_shapes
: f

cond/mod_1FloorModcond/strided_slice_1:output:0cond/add_1:z:0*
T0*
_output_shapes
: u
cond/crop_to_bounding_box/ShapeShape&cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 b
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: h
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 d
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: j
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*1
_output_shapes
:???????????c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :?f
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*1
_output_shapes
:????????????
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?g
?
cond_true_16671;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/Assert?$cond/stateful_uniform/RngReadAndSkipe
cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:[
cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : _
cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????e
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
cond/stateful_uniform/ProdProd$cond/stateful_uniform/shape:output:0$cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ^
cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :y
cond/stateful_uniform/Cast_1Cast#cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$cond/stateful_uniform/RngReadAndSkipRngReadAndSkip-cond_stateful_uniform_rngreadandskip_resource%cond/stateful_uniform/Cast/x:output:0 cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:s
)cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
cond/stateful_uniform/BitcastBitcast,cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0u
+cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
cond/stateful_uniformStatelessRandomUniformIntV2$cond/stateful_uniform/shape:output:0(cond/stateful_uniform/Bitcast_1:output:0&cond/stateful_uniform/Bitcast:output:0"cond/stateful_uniform/alg:output:0"cond/stateful_uniform/min:output:0"cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0b
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/stateful_uniform:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :U
cond/addAddV2cond_add_subcond/add/y:output:0*
T0*
_output_shapes
: `
cond/modFloorModcond/strided_slice:output:0cond/add:z:0*
T0*
_output_shapes
: d
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :]

cond/add_1AddV2cond_add_1_sub_1cond/add_1/y:output:0*
T0*
_output_shapes
: f

cond/mod_1FloorModcond/strided_slice_1:output:0cond/add_1:z:0*
T0*
_output_shapes
: u
cond/crop_to_bounding_box/ShapeShape&cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 b
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: h
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 d
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: j
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*1
_output_shapes
:???????????c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :?f
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*1
_output_shapes
:????????????
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
G__inference_block5_conv2_layer_call_and_return_conditional_losses_13236

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
G__inference_block2_conv1_layer_call_and_return_conditional_losses_17101

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????pp?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????pp?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????pp@
 
_user_specified_nameinputs
?
?
,__inference_sequential_2_layer_call_fn_15507

inputs
unknown:	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12931y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_13045

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?&
?	
B__inference_model_3_layer_call_and_return_conditional_losses_14357
input_8!
tf_nn_bias_add_3_biasadd_bias 
sequential_2_14294:	%
vgg16_14297:@
vgg16_14299:@%
vgg16_14301:@@
vgg16_14303:@&
vgg16_14305:@?
vgg16_14307:	?'
vgg16_14309:??
vgg16_14311:	?'
vgg16_14313:??
vgg16_14315:	?'
vgg16_14317:??
vgg16_14319:	?'
vgg16_14321:??
vgg16_14323:	?'
vgg16_14325:??
vgg16_14327:	?'
vgg16_14329:??
vgg16_14331:	?'
vgg16_14333:??
vgg16_14335:	?'
vgg16_14337:??
vgg16_14339:	?'
vgg16_14341:??
vgg16_14343:	?'
vgg16_14345:??
vgg16_14347:	? 
dense_3_14351:	?
dense_3_14353:
identity??dense_3/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?vgg16/StatefulPartitionedCall
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ?????
(tf.__operators__.getitem_3/strided_sliceStridedSliceinput_87tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask?
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:????????????
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall!tf.nn.bias_add_3/BiasAdd:output:0sequential_2_14294*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12665?
vgg16/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0vgg16_14297vgg16_14299vgg16_14301vgg16_14303vgg16_14305vgg16_14307vgg16_14309vgg16_14311vgg16_14313vgg16_14315vgg16_14317vgg16_14319vgg16_14321vgg16_14323vgg16_14325vgg16_14327vgg16_14329vgg16_14331vgg16_14333vgg16_14335vgg16_14337vgg16_14339vgg16_14341vgg16_14343vgg16_14345vgg16_14347*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_13261?
*global_average_pooling2d_3/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_13849?
dense_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0dense_3_14351dense_3_14353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_13932w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_3/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_8: 

_output_shapes
:
?	
?
B__inference_dense_3_layer_call_and_return_conditional_losses_16486

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
2model_3_sequential_2_random_crop_2_cond_true_12101^
Pmodel_3_sequential_2_random_crop_2_cond_stateful_uniform_rngreadandskip_resource:	V
Rmodel_3_sequential_2_random_crop_2_cond_add_model_3_sequential_2_random_crop_2_subZ
Vmodel_3_sequential_2_random_crop_2_cond_add_1_model_3_sequential_2_random_crop_2_sub_1w
smodel_3_sequential_2_random_crop_2_cond_crop_to_bounding_box_shape_model_3_sequential_2_center_crop_3_cond_identity4
0model_3_sequential_2_random_crop_2_cond_identity??Jmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert?Lmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert?Lmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert?Lmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert?Gmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip?
>model_3/sequential_2/random_crop_2/cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:~
<model_3/sequential_2/random_crop_2/cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : ?
<model_3/sequential_2/random_crop_2/cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
>model_3/sequential_2/random_crop_2/cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
=model_3/sequential_2/random_crop_2/cond/stateful_uniform/ProdProdGmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/shape:output:0Gmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ?
?model_3/sequential_2/random_crop_2/cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
?model_3/sequential_2/random_crop_2/cond/stateful_uniform/Cast_1CastFmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
Gmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkipRngReadAndSkipPmodel_3_sequential_2_random_crop_2_cond_stateful_uniform_rngreadandskip_resourceHmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/Cast/x:output:0Cmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:?
Lmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Nmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Nmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Fmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_sliceStridedSliceOmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip:value:0Umodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack:output:0Wmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_1:output:0Wmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
@model_3/sequential_2/random_crop_2/cond/stateful_uniform/BitcastBitcastOmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
Nmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Pmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Pmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1StridedSliceOmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip:value:0Wmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack:output:0Ymodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_1:output:0Ymodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
Bmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/Bitcast_1BitcastQmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0~
<model_3/sequential_2/random_crop_2/cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
8model_3/sequential_2/random_crop_2/cond/stateful_uniformStatelessRandomUniformIntV2Gmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/shape:output:0Kmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/Bitcast_1:output:0Imodel_3/sequential_2/random_crop_2/cond/stateful_uniform/Bitcast:output:0Emodel_3/sequential_2/random_crop_2/cond/stateful_uniform/alg:output:0Emodel_3/sequential_2/random_crop_2/cond/stateful_uniform/min:output:0Emodel_3/sequential_2/random_crop_2/cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0?
;model_3/sequential_2/random_crop_2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=model_3/sequential_2/random_crop_2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=model_3/sequential_2/random_crop_2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5model_3/sequential_2/random_crop_2/cond/strided_sliceStridedSliceAmodel_3/sequential_2/random_crop_2/cond/stateful_uniform:output:0Dmodel_3/sequential_2/random_crop_2/cond/strided_slice/stack:output:0Fmodel_3/sequential_2/random_crop_2/cond/strided_slice/stack_1:output:0Fmodel_3/sequential_2/random_crop_2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-model_3/sequential_2/random_crop_2/cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
+model_3/sequential_2/random_crop_2/cond/addAddV2Rmodel_3_sequential_2_random_crop_2_cond_add_model_3_sequential_2_random_crop_2_sub6model_3/sequential_2/random_crop_2/cond/add/y:output:0*
T0*
_output_shapes
: ?
+model_3/sequential_2/random_crop_2/cond/modFloorMod>model_3/sequential_2/random_crop_2/cond/strided_slice:output:0/model_3/sequential_2/random_crop_2/cond/add:z:0*
T0*
_output_shapes
: ?
=model_3/sequential_2/random_crop_2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
?model_3/sequential_2/random_crop_2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?model_3/sequential_2/random_crop_2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7model_3/sequential_2/random_crop_2/cond/strided_slice_1StridedSliceAmodel_3/sequential_2/random_crop_2/cond/stateful_uniform:output:0Fmodel_3/sequential_2/random_crop_2/cond/strided_slice_1/stack:output:0Hmodel_3/sequential_2/random_crop_2/cond/strided_slice_1/stack_1:output:0Hmodel_3/sequential_2/random_crop_2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/model_3/sequential_2/random_crop_2/cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
-model_3/sequential_2/random_crop_2/cond/add_1AddV2Vmodel_3_sequential_2_random_crop_2_cond_add_1_model_3_sequential_2_random_crop_2_sub_18model_3/sequential_2/random_crop_2/cond/add_1/y:output:0*
T0*
_output_shapes
: ?
-model_3/sequential_2/random_crop_2/cond/mod_1FloorMod@model_3/sequential_2/random_crop_2/cond/strided_slice_1:output:01model_3/sequential_2/random_crop_2/cond/add_1:z:0*
T0*
_output_shapes
: ?
Bmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/ShapeShapesmodel_3_sequential_2_random_crop_2_cond_crop_to_bounding_box_shape_model_3_sequential_2_center_crop_3_cond_identity*
T0*
_output_shapes
:?
Dmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/unstackUnpackKmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num?
Kmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Imodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqualGreaterEqual1model_3/sequential_2/random_crop_2/cond/mod_1:z:0Tmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Imodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Qmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Jmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertAssertMmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual:z:0Zmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
Mmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Kmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual/model_3/sequential_2/random_crop_2/cond/mod:z:0Vmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
Kmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Smodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Lmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertAssertOmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/GreaterEqual_1:z:0\model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0K^model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 ?
Bmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
@model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/addAddV2Kmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/add/x:output:01model_3/sequential_2/random_crop_2/cond/mod_1:z:0*
T0*
_output_shapes
: ?
Hmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
Fmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual	LessEqualDmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/add:z:0Qmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
Kmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Smodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Lmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertAssertJmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual:z:0\model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0M^model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 ?
Dmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
Bmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/add_1AddV2Mmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/add_1/x:output:0/model_3/sequential_2/random_crop_2/cond/mod:z:0*
T0*
_output_shapes
: ?
Jmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
Hmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1	LessEqualFmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/add_1:z:0Smodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
Kmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Smodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Lmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/AssertAssertLmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/LessEqual_1:z:0\model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0M^model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
Omodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependencyIdentitysmodel_3_sequential_2_random_crop_2_cond_crop_to_bounding_box_shape_model_3_sequential_2_center_crop_3_cond_identityK^model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertM^model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertM^model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertM^model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert*
T0*C
_class9
75loc:@model_3/sequential_2/center_crop_3/cond/Identity*1
_output_shapes
:????????????
Dmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
Bmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stackPackMmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stack/0:output:0/model_3/sequential_2/random_crop_2/cond/mod:z:01model_3/sequential_2/random_crop_2/cond/mod_1:z:0Mmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
Dmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_1ShapeXmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Pmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Rmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Rmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_sliceStridedSliceMmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_1:output:0Ymodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack:output:0[model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_1:output:0[model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Dmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_2ShapeXmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Rmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Tmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Tmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Lmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1StridedSliceMmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Shape_2:output:0[model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack:output:0]model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0]model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Fmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :??
Fmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
Dmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1PackSmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice:output:0Omodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/1:output:0Omodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1/2:output:0Umodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
Bmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/SliceSliceXmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/control_dependency:output:0Kmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stack:output:0Mmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
0model_3/sequential_2/random_crop_2/cond/IdentityIdentityKmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Slice:output:0-^model_3/sequential_2/random_crop_2/cond/NoOp*
T0*1
_output_shapes
:????????????
,model_3/sequential_2/random_crop_2/cond/NoOpNoOpK^model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertM^model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertM^model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertM^model_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/AssertH^model_3/sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "m
0model_3_sequential_2_random_crop_2_cond_identity9model_3/sequential_2/random_crop_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????2?
Jmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/AssertJmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert/Assert2?
Lmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/AssertLmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_1/Assert2?
Lmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/AssertLmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_2/Assert2?
Lmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/AssertLmodel_3/sequential_2/random_crop_2/cond/crop_to_bounding_box/Assert_3/Assert2?
Gmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkipGmodel_3/sequential_2/random_crop_2/cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?'
j
cond_false_12377
cond_placeholder
cond_placeholder_1
cond_shape_inputs
cond_identityK

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:k
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????m
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/Shape:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM

cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?d
cond/mulMulcond/strided_slice_1:output:0cond/mul/y:output:0*
T0*
_output_shapes
: O
	cond/CastCastcond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: S
cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C`
cond/truedivRealDivcond/Cast:y:0cond/truediv/y:output:0*
T0*
_output_shapes
: U
cond/Cast_1Castcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: O
cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :?f

cond/mul_1Mulcond/strided_slice:output:0cond/mul_1/y:output:0*
T0*
_output_shapes
: S
cond/Cast_2Castcond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `Cf
cond/truediv_1RealDivcond/Cast_2:y:0cond/truediv_1/y:output:0*
T0*
_output_shapes
: W
cond/Cast_3Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
cond/MinimumMinimumcond/strided_slice:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: j
cond/Minimum_1Minimumcond/strided_slice_1:output:0cond/Cast_3:y:0*
T0*
_output_shapes
: _
cond/subSubcond/strided_slice:output:0cond/Minimum:z:0*
T0*
_output_shapes
: Q
cond/Cast_4Castcond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_2RealDivcond/Cast_4:y:0cond/truediv_2/y:output:0*
T0*
_output_shapes
: W
cond/Cast_5Castcond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: e

cond/sub_1Subcond/strided_slice_1:output:0cond/Minimum_1:z:0*
T0*
_output_shapes
: S
cond/Cast_6Castcond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: U
cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
cond/truediv_3RealDivcond/Cast_6:y:0cond/truediv_3/y:output:0*
T0*
_output_shapes
: W
cond/Cast_7Castcond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: N
cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : N
cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?

cond/stackPackcond/stack/0:output:0cond/Cast_5:y:0cond/Cast_7:y:0cond/stack/3:output:0*
N*
T0*
_output_shapes
:Y
cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????Y
cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
cond/stack_1Packcond/stack_1/0:output:0cond/Minimum:z:0cond/Minimum_1:z:0cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?

cond/SliceSlicecond_shape_inputscond/stack:output:0cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????a
cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
cond/resize/ResizeBilinearResizeBilinearcond/Slice:output:0cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
cond/IdentityIdentity+cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
G__inference_block3_conv2_layer_call_and_return_conditional_losses_13132

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????88?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_14002
input_8
unknown
	unknown_0:	#
	unknown_1:@
	unknown_2:@#
	unknown_3:@@
	unknown_4:@$
	unknown_5:@?
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?&

unknown_25:??

unknown_26:	?

unknown_27:	?

unknown_28:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_13939o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_8: 

_output_shapes
:
?=
?
+sequential_2_center_crop_3_cond_false_14603/
+sequential_2_center_crop_3_cond_placeholder1
-sequential_2_center_crop_3_cond_placeholder_1B
>sequential_2_center_crop_3_cond_shape_tf_nn_bias_add_3_biasadd,
(sequential_2_center_crop_3_cond_identity?
%sequential_2/center_crop_3/cond/ShapeShape>sequential_2_center_crop_3_cond_shape_tf_nn_bias_add_3_biasadd*
T0*
_output_shapes
:?
3sequential_2/center_crop_3/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
5sequential_2/center_crop_3/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????
5sequential_2/center_crop_3/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-sequential_2/center_crop_3/cond/strided_sliceStridedSlice.sequential_2/center_crop_3/cond/Shape:output:0<sequential_2/center_crop_3/cond/strided_slice/stack:output:0>sequential_2/center_crop_3/cond/strided_slice/stack_1:output:0>sequential_2/center_crop_3/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_2/center_crop_3/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
7sequential_2/center_crop_3/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
7sequential_2/center_crop_3/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_2/center_crop_3/cond/strided_slice_1StridedSlice.sequential_2/center_crop_3/cond/Shape:output:0>sequential_2/center_crop_3/cond/strided_slice_1/stack:output:0@sequential_2/center_crop_3/cond/strided_slice_1/stack_1:output:0@sequential_2/center_crop_3/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
%sequential_2/center_crop_3/cond/mul/yConst*
_output_shapes
: *
dtype0*
value
B :??
#sequential_2/center_crop_3/cond/mulMul8sequential_2/center_crop_3/cond/strided_slice_1:output:0.sequential_2/center_crop_3/cond/mul/y:output:0*
T0*
_output_shapes
: ?
$sequential_2/center_crop_3/cond/CastCast'sequential_2/center_crop_3/cond/mul:z:0*

DstT0*

SrcT0*
_output_shapes
: n
)sequential_2/center_crop_3/cond/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
'sequential_2/center_crop_3/cond/truedivRealDiv(sequential_2/center_crop_3/cond/Cast:y:02sequential_2/center_crop_3/cond/truediv/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_1Cast+sequential_2/center_crop_3/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: j
'sequential_2/center_crop_3/cond/mul_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
%sequential_2/center_crop_3/cond/mul_1Mul6sequential_2/center_crop_3/cond/strided_slice:output:00sequential_2/center_crop_3/cond/mul_1/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_2Cast)sequential_2/center_crop_3/cond/mul_1:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/center_crop_3/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  `C?
)sequential_2/center_crop_3/cond/truediv_1RealDiv*sequential_2/center_crop_3/cond/Cast_2:y:04sequential_2/center_crop_3/cond/truediv_1/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_3Cast-sequential_2/center_crop_3/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
'sequential_2/center_crop_3/cond/MinimumMinimum6sequential_2/center_crop_3/cond/strided_slice:output:0*sequential_2/center_crop_3/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
)sequential_2/center_crop_3/cond/Minimum_1Minimum8sequential_2/center_crop_3/cond/strided_slice_1:output:0*sequential_2/center_crop_3/cond/Cast_3:y:0*
T0*
_output_shapes
: ?
#sequential_2/center_crop_3/cond/subSub6sequential_2/center_crop_3/cond/strided_slice:output:0+sequential_2/center_crop_3/cond/Minimum:z:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_4Cast'sequential_2/center_crop_3/cond/sub:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/center_crop_3/cond/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
)sequential_2/center_crop_3/cond/truediv_2RealDiv*sequential_2/center_crop_3/cond/Cast_4:y:04sequential_2/center_crop_3/cond/truediv_2/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_5Cast-sequential_2/center_crop_3/cond/truediv_2:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
%sequential_2/center_crop_3/cond/sub_1Sub8sequential_2/center_crop_3/cond/strided_slice_1:output:0-sequential_2/center_crop_3/cond/Minimum_1:z:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_6Cast)sequential_2/center_crop_3/cond/sub_1:z:0*

DstT0*

SrcT0*
_output_shapes
: p
+sequential_2/center_crop_3/cond/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
)sequential_2/center_crop_3/cond/truediv_3RealDiv*sequential_2/center_crop_3/cond/Cast_6:y:04sequential_2/center_crop_3/cond/truediv_3/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_7Cast-sequential_2/center_crop_3/cond/truediv_3:z:0*

DstT0*

SrcT0*
_output_shapes
: i
'sequential_2/center_crop_3/cond/stack/0Const*
_output_shapes
: *
dtype0*
value	B : i
'sequential_2/center_crop_3/cond/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
%sequential_2/center_crop_3/cond/stackPack0sequential_2/center_crop_3/cond/stack/0:output:0*sequential_2/center_crop_3/cond/Cast_5:y:0*sequential_2/center_crop_3/cond/Cast_7:y:00sequential_2/center_crop_3/cond/stack/3:output:0*
N*
T0*
_output_shapes
:t
)sequential_2/center_crop_3/cond/stack_1/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????t
)sequential_2/center_crop_3/cond/stack_1/3Const*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_2/center_crop_3/cond/stack_1Pack2sequential_2/center_crop_3/cond/stack_1/0:output:0+sequential_2/center_crop_3/cond/Minimum:z:0-sequential_2/center_crop_3/cond/Minimum_1:z:02sequential_2/center_crop_3/cond/stack_1/3:output:0*
N*
T0*
_output_shapes
:?
%sequential_2/center_crop_3/cond/SliceSlice>sequential_2_center_crop_3_cond_shape_tf_nn_bias_add_3_biasadd.sequential_2/center_crop_3/cond/stack:output:00sequential_2/center_crop_3/cond/stack_1:output:0*
Index0*
T0*:
_output_shapes(
&:$????????????????????|
+sequential_2/center_crop_3/cond/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
5sequential_2/center_crop_3/cond/resize/ResizeBilinearResizeBilinear.sequential_2/center_crop_3/cond/Slice:output:04sequential_2/center_crop_3/cond/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
(sequential_2/center_crop_3/cond/IdentityIdentityFsequential_2/center_crop_3/cond/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"]
(sequential_2_center_crop_3_cond_identity1sequential_2/center_crop_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
G__inference_block3_conv1_layer_call_and_return_conditional_losses_13115

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????88?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?I
?
cond_true_16518
cond_truediv_cast_sub
cond_truediv_1_cast_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/AssertP
cond/truediv/yConst*
_output_shapes
: *
dtype0*
value	B :`
cond/truediv/CastCastcond_truediv_cast_sub*

DstT0*

SrcT0*
_output_shapes
: d
cond/truediv/Cast_1Castcond/truediv/y:output:0*

DstT0*

SrcT0*
_output_shapes
: h
cond/truedivRealDivcond/truediv/Cast:y:0cond/truediv/Cast_1:y:0*
T0*
_output_shapes
: S
	cond/CastCastcond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: R
cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
value	B :f
cond/truediv_1/CastCastcond_truediv_1_cast_sub_1*

DstT0*

SrcT0*
_output_shapes
: h
cond/truediv_1/Cast_1Castcond/truediv_1/y:output:0*

DstT0*

SrcT0*
_output_shapes
: n
cond/truediv_1RealDivcond/truediv_1/Cast:y:0cond/truediv_1/Cast_1:y:0*
T0*
_output_shapes
: W
cond/Cast_1Castcond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: u
cond/crop_to_bounding_box/ShapeShape&cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/Cast_1:y:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/Cast:y:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 b
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/Cast_1:y:0*
T0*
_output_shapes
: h
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 d
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/Cast:y:0*
T0*
_output_shapes
: j
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*1
_output_shapes
:???????????c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/Cast:y:0cond/Cast_1:y:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :?f
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*1
_output_shapes
:????????????
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
??
?
B__inference_model_3_layer_call_and_return_conditional_losses_14978

inputs!
tf_nn_bias_add_3_biasadd_bias5
'sequential_2_random_crop_2_cond_input_0:	K
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@?A
2vgg16_block2_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block2_conv2_conv2d_readvariableop_resource:??A
2vgg16_block2_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv1_conv2d_readvariableop_resource:??A
2vgg16_block3_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv2_conv2d_readvariableop_resource:??A
2vgg16_block3_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block3_conv3_conv2d_readvariableop_resource:??A
2vgg16_block3_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv1_conv2d_readvariableop_resource:??A
2vgg16_block4_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv2_conv2d_readvariableop_resource:??A
2vgg16_block4_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block4_conv3_conv2d_readvariableop_resource:??A
2vgg16_block4_conv3_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv1_conv2d_readvariableop_resource:??A
2vgg16_block5_conv1_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv2_conv2d_readvariableop_resource:??A
2vgg16_block5_conv2_biasadd_readvariableop_resource:	?M
1vgg16_block5_conv3_conv2d_readvariableop_resource:??A
2vgg16_block5_conv3_biasadd_readvariableop_resource:	?9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?sequential_2/center_crop_3/cond?sequential_2/random_crop_2/cond?)vgg16/block1_conv1/BiasAdd/ReadVariableOp?(vgg16/block1_conv1/Conv2D/ReadVariableOp?)vgg16/block1_conv2/BiasAdd/ReadVariableOp?(vgg16/block1_conv2/Conv2D/ReadVariableOp?)vgg16/block2_conv1/BiasAdd/ReadVariableOp?(vgg16/block2_conv1/Conv2D/ReadVariableOp?)vgg16/block2_conv2/BiasAdd/ReadVariableOp?(vgg16/block2_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv1/BiasAdd/ReadVariableOp?(vgg16/block3_conv1/Conv2D/ReadVariableOp?)vgg16/block3_conv2/BiasAdd/ReadVariableOp?(vgg16/block3_conv2/Conv2D/ReadVariableOp?)vgg16/block3_conv3/BiasAdd/ReadVariableOp?(vgg16/block3_conv3/Conv2D/ReadVariableOp?)vgg16/block4_conv1/BiasAdd/ReadVariableOp?(vgg16/block4_conv1/Conv2D/ReadVariableOp?)vgg16/block4_conv2/BiasAdd/ReadVariableOp?(vgg16/block4_conv2/Conv2D/ReadVariableOp?)vgg16/block4_conv3/BiasAdd/ReadVariableOp?(vgg16/block4_conv3/Conv2D/ReadVariableOp?)vgg16/block5_conv1/BiasAdd/ReadVariableOp?(vgg16/block5_conv1/Conv2D/ReadVariableOp?)vgg16/block5_conv2/BiasAdd/ReadVariableOp?(vgg16/block5_conv2/Conv2D/ReadVariableOp?)vgg16/block5_conv3/BiasAdd/ReadVariableOp?(vgg16/block5_conv3/Conv2D/ReadVariableOp
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ?????
(tf.__operators__.getitem_3/strided_sliceStridedSliceinputs7tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask?
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:???????????q
 sequential_2/center_crop_3/ShapeShape!tf.nn.bias_add_3/BiasAdd:output:0*
T0*
_output_shapes
:?
.sequential_2/center_crop_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
0sequential_2/center_crop_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????z
0sequential_2/center_crop_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_2/center_crop_3/strided_sliceStridedSlice)sequential_2/center_crop_3/Shape:output:07sequential_2/center_crop_3/strided_slice/stack:output:09sequential_2/center_crop_3/strided_slice/stack_1:output:09sequential_2/center_crop_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 sequential_2/center_crop_3/sub/yConst*
_output_shapes
: *
dtype0*
value
B :??
sequential_2/center_crop_3/subSub1sequential_2/center_crop_3/strided_slice:output:0)sequential_2/center_crop_3/sub/y:output:0*
T0*
_output_shapes
: ?
0sequential_2/center_crop_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
2sequential_2/center_crop_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????|
2sequential_2/center_crop_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_2/center_crop_3/strided_slice_1StridedSlice)sequential_2/center_crop_3/Shape:output:09sequential_2/center_crop_3/strided_slice_1/stack:output:0;sequential_2/center_crop_3/strided_slice_1/stack_1:output:0;sequential_2/center_crop_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"sequential_2/center_crop_3/sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
 sequential_2/center_crop_3/sub_1Sub3sequential_2/center_crop_3/strided_slice_1:output:0+sequential_2/center_crop_3/sub_1/y:output:0*
T0*
_output_shapes
: k
)sequential_2/center_crop_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_2/center_crop_3/GreaterEqualGreaterEqual"sequential_2/center_crop_3/sub:z:02sequential_2/center_crop_3/GreaterEqual/y:output:0*
T0*
_output_shapes
: m
+sequential_2/center_crop_3/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_2/center_crop_3/GreaterEqual_1GreaterEqual$sequential_2/center_crop_3/sub_1:z:04sequential_2/center_crop_3/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/Rank/packedPack+sequential_2/center_crop_3/GreaterEqual:z:0-sequential_2/center_crop_3/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:a
sequential_2/center_crop_3/RankConst*
_output_shapes
: *
dtype0*
value	B :h
&sequential_2/center_crop_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : h
&sequential_2/center_crop_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
 sequential_2/center_crop_3/rangeRange/sequential_2/center_crop_3/range/start:output:0(sequential_2/center_crop_3/Rank:output:0/sequential_2/center_crop_3/range/delta:output:0*
_output_shapes
:?
$sequential_2/center_crop_3/All/inputPack+sequential_2/center_crop_3/GreaterEqual:z:0-sequential_2/center_crop_3/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:?
sequential_2/center_crop_3/AllAll-sequential_2/center_crop_3/All/input:output:0)sequential_2/center_crop_3/range:output:0*
_output_shapes
: ?
sequential_2/center_crop_3/condIf'sequential_2/center_crop_3/All:output:0"sequential_2/center_crop_3/sub:z:0$sequential_2/center_crop_3/sub_1:z:0!tf.nn.bias_add_3/BiasAdd:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *>
else_branch/R-
+sequential_2_center_crop_3_cond_false_14603*0
output_shapes
:???????????*=
then_branch.R,
*sequential_2_center_crop_3_cond_true_14602?
(sequential_2/center_crop_3/cond/IdentityIdentity(sequential_2/center_crop_3/cond:output:0*
T0*1
_output_shapes
:????????????
 sequential_2/random_crop_2/ShapeShape1sequential_2/center_crop_3/cond/Identity:output:0*
T0*
_output_shapes
:?
.sequential_2/random_crop_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
0sequential_2/random_crop_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????z
0sequential_2/random_crop_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_2/random_crop_2/strided_sliceStridedSlice)sequential_2/random_crop_2/Shape:output:07sequential_2/random_crop_2/strided_slice/stack:output:09sequential_2/random_crop_2/strided_slice/stack_1:output:09sequential_2/random_crop_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 sequential_2/random_crop_2/sub/yConst*
_output_shapes
: *
dtype0*
value
B :??
sequential_2/random_crop_2/subSub1sequential_2/random_crop_2/strided_slice:output:0)sequential_2/random_crop_2/sub/y:output:0*
T0*
_output_shapes
: ?
0sequential_2/random_crop_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
2sequential_2/random_crop_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????|
2sequential_2/random_crop_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_2/random_crop_2/strided_slice_1StridedSlice)sequential_2/random_crop_2/Shape:output:09sequential_2/random_crop_2/strided_slice_1/stack:output:0;sequential_2/random_crop_2/strided_slice_1/stack_1:output:0;sequential_2/random_crop_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"sequential_2/random_crop_2/sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
 sequential_2/random_crop_2/sub_1Sub3sequential_2/random_crop_2/strided_slice_1:output:0+sequential_2/random_crop_2/sub_1/y:output:0*
T0*
_output_shapes
: k
)sequential_2/random_crop_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
'sequential_2/random_crop_2/GreaterEqualGreaterEqual"sequential_2/random_crop_2/sub:z:02sequential_2/random_crop_2/GreaterEqual/y:output:0*
T0*
_output_shapes
: m
+sequential_2/random_crop_2/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
)sequential_2/random_crop_2/GreaterEqual_1GreaterEqual$sequential_2/random_crop_2/sub_1:z:04sequential_2/random_crop_2/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: j
(sequential_2/random_crop_2/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
&sequential_2/random_crop_2/Rank/packedPack1sequential_2/random_crop_2/Rank/packed/0:output:0+sequential_2/random_crop_2/GreaterEqual:z:0-sequential_2/random_crop_2/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:a
sequential_2/random_crop_2/RankConst*
_output_shapes
: *
dtype0*
value	B :h
&sequential_2/random_crop_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : h
&sequential_2/random_crop_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
 sequential_2/random_crop_2/rangeRange/sequential_2/random_crop_2/range/start:output:0(sequential_2/random_crop_2/Rank:output:0/sequential_2/random_crop_2/range/delta:output:0*
_output_shapes
:h
&sequential_2/random_crop_2/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
$sequential_2/random_crop_2/All/inputPack/sequential_2/random_crop_2/All/input/0:output:0+sequential_2/random_crop_2/GreaterEqual:z:0-sequential_2/random_crop_2/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:?
sequential_2/random_crop_2/AllAll-sequential_2/random_crop_2/All/input:output:0)sequential_2/random_crop_2/range:output:0*
_output_shapes
: ?
sequential_2/random_crop_2/condIf'sequential_2/random_crop_2/All:output:0'sequential_2_random_crop_2_cond_input_0"sequential_2/random_crop_2/sub:z:0$sequential_2/random_crop_2/sub_1:z:01sequential_2/center_crop_3/cond/Identity:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *>
else_branch/R-
+sequential_2_random_crop_2_cond_false_14738*0
output_shapes
:???????????*=
then_branch.R,
*sequential_2_random_crop_2_cond_true_14737?
(sequential_2/random_crop_2/cond/IdentityIdentity(sequential_2/random_crop_2/cond:output:0*
T0*1
_output_shapes
:????????????
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
vgg16/block1_conv1/Conv2DConv2D1sequential_2/random_crop_2/cond/Identity:output:00vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
?
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
?
)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp??
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
?
)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp??
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
?
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
1global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
global_average_pooling2d_3/MeanMean"vgg16/block5_pool/MaxPool:output:0:global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMul(global_average_pooling2d_3/Mean:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????

NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp ^sequential_2/center_crop_3/cond ^sequential_2/random_crop_2/cond*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2B
sequential_2/center_crop_3/condsequential_2/center_crop_3/cond2B
sequential_2/random_crop_2/condsequential_2/random_crop_2/cond2V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?
?
,__inference_block3_conv1_layer_call_fn_17140

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_13115x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????88?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12665

inputs!
random_crop_2_12655:	
identity??%center_crop_3/StatefulPartitionedCall?%random_crop_2/StatefulPartitionedCall?
%center_crop_3/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_center_crop_3_layer_call_and_return_conditional_losses_12486?
%random_crop_2/StatefulPartitionedCallStatefulPartitionedCall.center_crop_3/StatefulPartitionedCall:output:0random_crop_2_12655*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_random_crop_2_layer_call_and_return_conditional_losses_12654?
!random_contrast_2/PartitionedCallPartitionedCall.random_crop_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_12662?
IdentityIdentity*random_contrast_2/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp&^center_crop_3/StatefulPartitionedCall&^random_crop_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2N
%center_crop_3/StatefulPartitionedCall%center_crop_3/StatefulPartitionedCall2N
%random_crop_2/StatefulPartitionedCall%random_crop_2/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?V
?
@__inference_vgg16_layer_call_and_return_conditional_losses_13579

inputs,
block1_conv1_13508:@ 
block1_conv1_13510:@,
block1_conv2_13513:@@ 
block1_conv2_13515:@-
block2_conv1_13519:@?!
block2_conv1_13521:	?.
block2_conv2_13524:??!
block2_conv2_13526:	?.
block3_conv1_13530:??!
block3_conv1_13532:	?.
block3_conv2_13535:??!
block3_conv2_13537:	?.
block3_conv3_13540:??!
block3_conv3_13542:	?.
block4_conv1_13546:??!
block4_conv1_13548:	?.
block4_conv2_13551:??!
block4_conv2_13553:	?.
block4_conv3_13556:??!
block4_conv3_13558:	?.
block5_conv1_13562:??!
block5_conv1_13564:	?.
block5_conv2_13567:??!
block5_conv2_13569:	?.
block5_conv3_13572:??!
block5_conv3_13574:	?
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?$block5_conv2/StatefulPartitionedCall?$block5_conv3/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_13508block1_conv1_13510*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_13045?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_13513block1_conv2_13515*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_13062?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_12976?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_13519block2_conv1_13521*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_13080?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_13524block2_conv2_13526*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????pp?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_13097?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_12988?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_13530block3_conv1_13532*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_13115?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_13535block3_conv2_13537*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_13132?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_13540block3_conv3_13542*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????88?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_13149?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_13000?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_13546block4_conv1_13548*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_13167?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_13551block4_conv2_13553*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_13184?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_13556block4_conv3_13558*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_13201?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_13012?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_13562block5_conv1_13564*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_13219?
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_13567block5_conv2_13569*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_13236?
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_13572block5_conv3_13574*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_13253?
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_13024|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?g
?
cond_true_12517;
-cond_stateful_uniform_rngreadandskip_resource:	
cond_add_sub
cond_add_1_sub_1*
&cond_crop_to_bounding_box_shape_inputs
cond_identity??'cond/crop_to_bounding_box/Assert/Assert?)cond/crop_to_bounding_box/Assert_1/Assert?)cond/crop_to_bounding_box/Assert_2/Assert?)cond/crop_to_bounding_box/Assert_3/Assert?$cond/stateful_uniform/RngReadAndSkipe
cond/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:[
cond/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : _
cond/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :????e
cond/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
cond/stateful_uniform/ProdProd$cond/stateful_uniform/shape:output:0$cond/stateful_uniform/Const:output:0*
T0*
_output_shapes
: ^
cond/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :y
cond/stateful_uniform/Cast_1Cast#cond/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
$cond/stateful_uniform/RngReadAndSkipRngReadAndSkip-cond_stateful_uniform_rngreadandskip_resource%cond/stateful_uniform/Cast/x:output:0 cond/stateful_uniform/Cast_1:y:0*
_output_shapes
:s
)cond/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+cond/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+cond/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#cond/stateful_uniform/strided_sliceStridedSlice,cond/stateful_uniform/RngReadAndSkip:value:02cond/stateful_uniform/strided_slice/stack:output:04cond/stateful_uniform/strided_slice/stack_1:output:04cond/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
cond/stateful_uniform/BitcastBitcast,cond/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0u
+cond/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-cond/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%cond/stateful_uniform/strided_slice_1StridedSlice,cond/stateful_uniform/RngReadAndSkip:value:04cond/stateful_uniform/strided_slice_1/stack:output:06cond/stateful_uniform/strided_slice_1/stack_1:output:06cond/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
cond/stateful_uniform/Bitcast_1Bitcast.cond/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0[
cond/stateful_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
cond/stateful_uniformStatelessRandomUniformIntV2$cond/stateful_uniform/shape:output:0(cond/stateful_uniform/Bitcast_1:output:0&cond/stateful_uniform/Bitcast:output:0"cond/stateful_uniform/alg:output:0"cond/stateful_uniform/min:output:0"cond/stateful_uniform/max:output:0*
_output_shapes
:*
dtype0b
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: d
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:d
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_sliceStridedSlicecond/stateful_uniform:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

cond/add/yConst*
_output_shapes
: *
dtype0*
value	B :U
cond/addAddV2cond_add_subcond/add/y:output:0*
T0*
_output_shapes
: `
cond/modFloorModcond/strided_slice:output:0cond/add:z:0*
T0*
_output_shapes
: d
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
cond/strided_slice_1StridedSlicecond/stateful_uniform:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
cond/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :]

cond/add_1AddV2cond_add_1_sub_1cond/add_1/y:output:0*
T0*
_output_shapes
: f

cond/mod_1FloorModcond/strided_slice_1:output:0cond/add_1:z:0*
T0*
_output_shapes
: u
cond/crop_to_bounding_box/ShapeShape&cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/unstackUnpack(cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numj
(cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
&cond/crop_to_bounding_box/GreaterEqualGreaterEqualcond/mod_1:z:01cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
&cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
.cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
'cond/crop_to_bounding_box/Assert/AssertAssert*cond/crop_to_bounding_box/GreaterEqual:z:07cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 l
*cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
(cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcond/mod:z:03cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
0cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
)cond/crop_to_bounding_box/Assert_1/AssertAssert,cond/crop_to_bounding_box/GreaterEqual_1:z:09cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0(^cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 b
cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/addAddV2(cond/crop_to_bounding_box/add/x:output:0cond/mod_1:z:0*
T0*
_output_shapes
: h
%cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
#cond/crop_to_bounding_box/LessEqual	LessEqual!cond/crop_to_bounding_box/add:z:0.cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_2/AssertAssert'cond/crop_to_bounding_box/LessEqual:z:09cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 d
!cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
cond/crop_to_bounding_box/add_1AddV2*cond/crop_to_bounding_box/add_1/x:output:0cond/mod:z:0*
T0*
_output_shapes
: j
'cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
%cond/crop_to_bounding_box/LessEqual_1	LessEqual#cond/crop_to_bounding_box/add_1:z:00cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
(cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
0cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
)cond/crop_to_bounding_box/Assert_3/AssertAssert)cond/crop_to_bounding_box/LessEqual_1:z:09cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0*^cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
,cond/crop_to_bounding_box/control_dependencyIdentity&cond_crop_to_bounding_box_shape_inputs(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*1
_output_shapes
:???????????c
!cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : c
!cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
cond/crop_to_bounding_box/stackPack*cond/crop_to_bounding_box/stack/0:output:0cond/mod:z:0cond/mod_1:z:0*cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
!cond/crop_to_bounding_box/Shape_1Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:w
-cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'cond/crop_to_bounding_box/strided_sliceStridedSlice*cond/crop_to_bounding_box/Shape_1:output:06cond/crop_to_bounding_box/strided_slice/stack:output:08cond/crop_to_bounding_box/strided_slice/stack_1:output:08cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
!cond/crop_to_bounding_box/Shape_2Shape5cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:y
/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)cond/crop_to_bounding_box/strided_slice_1StridedSlice*cond/crop_to_bounding_box/Shape_2:output:08cond/crop_to_bounding_box/strided_slice_1/stack:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0:cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :?f
#cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
!cond/crop_to_bounding_box/stack_1Pack0cond/crop_to_bounding_box/strided_slice:output:0,cond/crop_to_bounding_box/stack_1/1:output:0,cond/crop_to_bounding_box/stack_1/2:output:02cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
cond/crop_to_bounding_box/SliceSlice5cond/crop_to_bounding_box/control_dependency:output:0(cond/crop_to_bounding_box/stack:output:0*cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
cond/IdentityIdentity(cond/crop_to_bounding_box/Slice:output:0
^cond/NoOp*
T0*1
_output_shapes
:????????????
	cond/NoOpNoOp(^cond/crop_to_bounding_box/Assert/Assert*^cond/crop_to_bounding_box/Assert_1/Assert*^cond/crop_to_bounding_box/Assert_2/Assert*^cond/crop_to_bounding_box/Assert_3/Assert%^cond/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :???????????2R
'cond/crop_to_bounding_box/Assert/Assert'cond/crop_to_bounding_box/Assert/Assert2V
)cond/crop_to_bounding_box/Assert_1/Assert)cond/crop_to_bounding_box/Assert_1/Assert2V
)cond/crop_to_bounding_box/Assert_2/Assert)cond/crop_to_bounding_box/Assert_2/Assert2V
)cond/crop_to_bounding_box/Assert_3/Assert)cond/crop_to_bounding_box/Assert_3/Assert2L
$cond/stateful_uniform/RngReadAndSkip$cond/stateful_uniform/RngReadAndSkip:

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?	
?
B__inference_dense_3_layer_call_and_return_conditional_losses_13932

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_block1_pool_layer_call_fn_17076

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_12976?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_model_3_layer_call_fn_14285
input_8
unknown
	unknown_0:	
	unknown_1:	#
	unknown_2:@
	unknown_3:@#
	unknown_4:@@
	unknown_5:@$
	unknown_6:@?
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?&

unknown_10:??

unknown_11:	?&

unknown_12:??

unknown_13:	?&

unknown_14:??

unknown_15:	?&

unknown_16:??

unknown_17:	?&

unknown_18:??

unknown_19:	?&

unknown_20:??

unknown_21:	?&

unknown_22:??

unknown_23:	?&

unknown_24:??

unknown_25:	?&

unknown_26:??

unknown_27:	?

unknown_28:	?

unknown_29:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_14153o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_8: 

_output_shapes
:
?h
?
*sequential_2_center_crop_3_cond_true_15011O
Ksequential_2_center_crop_3_cond_truediv_cast_sequential_2_center_crop_3_subS
Osequential_2_center_crop_3_cond_truediv_1_cast_sequential_2_center_crop_3_sub_1W
Ssequential_2_center_crop_3_cond_crop_to_bounding_box_shape_tf_nn_bias_add_3_biasadd,
(sequential_2_center_crop_3_cond_identity??Bsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert?Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert?Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert?Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assertk
)sequential_2/center_crop_3/cond/truediv/yConst*
_output_shapes
: *
dtype0*
value	B :?
,sequential_2/center_crop_3/cond/truediv/CastCastKsequential_2_center_crop_3_cond_truediv_cast_sequential_2_center_crop_3_sub*

DstT0*

SrcT0*
_output_shapes
: ?
.sequential_2/center_crop_3/cond/truediv/Cast_1Cast2sequential_2/center_crop_3/cond/truediv/y:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
'sequential_2/center_crop_3/cond/truedivRealDiv0sequential_2/center_crop_3/cond/truediv/Cast:y:02sequential_2/center_crop_3/cond/truediv/Cast_1:y:0*
T0*
_output_shapes
: ?
$sequential_2/center_crop_3/cond/CastCast+sequential_2/center_crop_3/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: m
+sequential_2/center_crop_3/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
.sequential_2/center_crop_3/cond/truediv_1/CastCastOsequential_2_center_crop_3_cond_truediv_1_cast_sequential_2_center_crop_3_sub_1*

DstT0*

SrcT0*
_output_shapes
: ?
0sequential_2/center_crop_3/cond/truediv_1/Cast_1Cast4sequential_2/center_crop_3/cond/truediv_1/y:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
)sequential_2/center_crop_3/cond/truediv_1RealDiv2sequential_2/center_crop_3/cond/truediv_1/Cast:y:04sequential_2/center_crop_3/cond/truediv_1/Cast_1:y:0*
T0*
_output_shapes
: ?
&sequential_2/center_crop_3/cond/Cast_1Cast-sequential_2/center_crop_3/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
:sequential_2/center_crop_3/cond/crop_to_bounding_box/ShapeShapeSsequential_2_center_crop_3_cond_crop_to_bounding_box_shape_tf_nn_bias_add_3_biasadd*
T0*
_output_shapes
:?
<sequential_2/center_crop_3/cond/crop_to_bounding_box/unstackUnpackCsequential_2/center_crop_3/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
num?
Csequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Asequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqualGreaterEqual*sequential_2/center_crop_3/cond/Cast_1:y:0Lsequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
Asequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Isequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
Bsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertAssertEsequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual:z:0Rsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
Esequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
Csequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqual(sequential_2/center_crop_3/cond/Cast:y:0Nsequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Ksequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertAssertGsequential_2/center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1:z:0Tsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:0C^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 }
:sequential_2/center_crop_3/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
8sequential_2/center_crop_3/cond/crop_to_bounding_box/addAddV2Csequential_2/center_crop_3/cond/crop_to_bounding_box/add/x:output:0*sequential_2/center_crop_3/cond/Cast_1:y:0*
T0*
_output_shapes
: ?
@sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
>sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual	LessEqual<sequential_2/center_crop_3/cond/crop_to_bounding_box/add:z:0Isequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Ksequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertAssertBsequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual:z:0Tsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:0E^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 
<sequential_2/center_crop_3/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
:sequential_2/center_crop_3/cond/crop_to_bounding_box/add_1AddV2Esequential_2/center_crop_3/cond/crop_to_bounding_box/add_1/x:output:0(sequential_2/center_crop_3/cond/Cast:y:0*
T0*
_output_shapes
: ?
Bsequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
@sequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1	LessEqual>sequential_2/center_crop_3/cond/crop_to_bounding_box/add_1:z:0Ksequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
Csequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Ksequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/AssertAssertDsequential_2/center_crop_3/cond/crop_to_bounding_box/LessEqual_1:z:0Tsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:0E^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
Gsequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependencyIdentitySsequential_2_center_crop_3_cond_crop_to_bounding_box_shape_tf_nn_bias_add_3_biasaddC^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert*
T0*+
_class!
loc:@tf.nn.bias_add_3/BiasAdd*1
_output_shapes
:???????????~
<sequential_2/center_crop_3/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : ~
<sequential_2/center_crop_3/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
:sequential_2/center_crop_3/cond/crop_to_bounding_box/stackPackEsequential_2/center_crop_3/cond/crop_to_bounding_box/stack/0:output:0(sequential_2/center_crop_3/cond/Cast:y:0*sequential_2/center_crop_3/cond/Cast_1:y:0Esequential_2/center_crop_3/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
<sequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_1ShapePsequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Hsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Jsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Jsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_sliceStridedSliceEsequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_1:output:0Qsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack:output:0Ssequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Ssequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<sequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_2ShapePsequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
Jsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Lsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Lsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1StridedSliceEsequential_2/center_crop_3/cond/crop_to_bounding_box/Shape_2:output:0Ssequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Usequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Usequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :??
>sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
<sequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1PackKsequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice:output:0Gsequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/1:output:0Gsequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1/2:output:0Msequential_2/center_crop_3/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
:sequential_2/center_crop_3/cond/crop_to_bounding_box/SliceSlicePsequential_2/center_crop_3/cond/crop_to_bounding_box/control_dependency:output:0Csequential_2/center_crop_3/cond/crop_to_bounding_box/stack:output:0Esequential_2/center_crop_3/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
(sequential_2/center_crop_3/cond/IdentityIdentityCsequential_2/center_crop_3/cond/crop_to_bounding_box/Slice:output:0%^sequential_2/center_crop_3/cond/NoOp*
T0*1
_output_shapes
:????????????
$sequential_2/center_crop_3/cond/NoOpNoOpC^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertE^sequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_2_center_crop_3_cond_identity1sequential_2/center_crop_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????2?
Bsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/AssertBsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert/Assert2?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertDsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert2?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertDsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert2?
Dsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/AssertDsequential_2/center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????
?
?
,__inference_block1_conv2_layer_call_fn_17060

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_13062y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_13000

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_vgg16_layer_call_fn_16256

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?%
	unknown_7:??
	unknown_8:	?%
	unknown_9:??

unknown_10:	?&

unknown_11:??

unknown_12:	?&

unknown_13:??

unknown_14:	?&

unknown_15:??

unknown_16:	?&

unknown_17:??

unknown_18:	?&

unknown_19:??

unknown_20:	?&

unknown_21:??

unknown_22:	?&

unknown_23:??

unknown_24:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_13579x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
я
?
 __inference__wrapped_model_12342
input_8)
%model_3_tf_nn_bias_add_3_biasadd_bias=
/model_3_sequential_2_random_crop_2_cond_input_0:	S
9model_3_vgg16_block1_conv1_conv2d_readvariableop_resource:@H
:model_3_vgg16_block1_conv1_biasadd_readvariableop_resource:@S
9model_3_vgg16_block1_conv2_conv2d_readvariableop_resource:@@H
:model_3_vgg16_block1_conv2_biasadd_readvariableop_resource:@T
9model_3_vgg16_block2_conv1_conv2d_readvariableop_resource:@?I
:model_3_vgg16_block2_conv1_biasadd_readvariableop_resource:	?U
9model_3_vgg16_block2_conv2_conv2d_readvariableop_resource:??I
:model_3_vgg16_block2_conv2_biasadd_readvariableop_resource:	?U
9model_3_vgg16_block3_conv1_conv2d_readvariableop_resource:??I
:model_3_vgg16_block3_conv1_biasadd_readvariableop_resource:	?U
9model_3_vgg16_block3_conv2_conv2d_readvariableop_resource:??I
:model_3_vgg16_block3_conv2_biasadd_readvariableop_resource:	?U
9model_3_vgg16_block3_conv3_conv2d_readvariableop_resource:??I
:model_3_vgg16_block3_conv3_biasadd_readvariableop_resource:	?U
9model_3_vgg16_block4_conv1_conv2d_readvariableop_resource:??I
:model_3_vgg16_block4_conv1_biasadd_readvariableop_resource:	?U
9model_3_vgg16_block4_conv2_conv2d_readvariableop_resource:??I
:model_3_vgg16_block4_conv2_biasadd_readvariableop_resource:	?U
9model_3_vgg16_block4_conv3_conv2d_readvariableop_resource:??I
:model_3_vgg16_block4_conv3_biasadd_readvariableop_resource:	?U
9model_3_vgg16_block5_conv1_conv2d_readvariableop_resource:??I
:model_3_vgg16_block5_conv1_biasadd_readvariableop_resource:	?U
9model_3_vgg16_block5_conv2_conv2d_readvariableop_resource:??I
:model_3_vgg16_block5_conv2_biasadd_readvariableop_resource:	?U
9model_3_vgg16_block5_conv3_conv2d_readvariableop_resource:??I
:model_3_vgg16_block5_conv3_biasadd_readvariableop_resource:	?A
.model_3_dense_3_matmul_readvariableop_resource:	?=
/model_3_dense_3_biasadd_readvariableop_resource:
identity??&model_3/dense_3/BiasAdd/ReadVariableOp?%model_3/dense_3/MatMul/ReadVariableOp?'model_3/sequential_2/center_crop_3/cond?'model_3/sequential_2/random_crop_2/cond?1model_3/vgg16/block1_conv1/BiasAdd/ReadVariableOp?0model_3/vgg16/block1_conv1/Conv2D/ReadVariableOp?1model_3/vgg16/block1_conv2/BiasAdd/ReadVariableOp?0model_3/vgg16/block1_conv2/Conv2D/ReadVariableOp?1model_3/vgg16/block2_conv1/BiasAdd/ReadVariableOp?0model_3/vgg16/block2_conv1/Conv2D/ReadVariableOp?1model_3/vgg16/block2_conv2/BiasAdd/ReadVariableOp?0model_3/vgg16/block2_conv2/Conv2D/ReadVariableOp?1model_3/vgg16/block3_conv1/BiasAdd/ReadVariableOp?0model_3/vgg16/block3_conv1/Conv2D/ReadVariableOp?1model_3/vgg16/block3_conv2/BiasAdd/ReadVariableOp?0model_3/vgg16/block3_conv2/Conv2D/ReadVariableOp?1model_3/vgg16/block3_conv3/BiasAdd/ReadVariableOp?0model_3/vgg16/block3_conv3/Conv2D/ReadVariableOp?1model_3/vgg16/block4_conv1/BiasAdd/ReadVariableOp?0model_3/vgg16/block4_conv1/Conv2D/ReadVariableOp?1model_3/vgg16/block4_conv2/BiasAdd/ReadVariableOp?0model_3/vgg16/block4_conv2/Conv2D/ReadVariableOp?1model_3/vgg16/block4_conv3/BiasAdd/ReadVariableOp?0model_3/vgg16/block4_conv3/Conv2D/ReadVariableOp?1model_3/vgg16/block5_conv1/BiasAdd/ReadVariableOp?0model_3/vgg16/block5_conv1/Conv2D/ReadVariableOp?1model_3/vgg16/block5_conv2/BiasAdd/ReadVariableOp?0model_3/vgg16/block5_conv2/Conv2D/ReadVariableOp?1model_3/vgg16/block5_conv3/BiasAdd/ReadVariableOp?0model_3/vgg16/block5_conv3/Conv2D/ReadVariableOp?
6model_3/tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
8model_3/tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
8model_3/tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ?????
0model_3/tf.__operators__.getitem_3/strided_sliceStridedSliceinput_8?model_3/tf.__operators__.getitem_3/strided_slice/stack:output:0Amodel_3/tf.__operators__.getitem_3/strided_slice/stack_1:output:0Amodel_3/tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask?
 model_3/tf.nn.bias_add_3/BiasAddBiasAdd9model_3/tf.__operators__.getitem_3/strided_slice:output:0%model_3_tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:????????????
(model_3/sequential_2/center_crop_3/ShapeShape)model_3/tf.nn.bias_add_3/BiasAdd:output:0*
T0*
_output_shapes
:?
6model_3/sequential_2/center_crop_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
8model_3/sequential_2/center_crop_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
8model_3/sequential_2/center_crop_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0model_3/sequential_2/center_crop_3/strided_sliceStridedSlice1model_3/sequential_2/center_crop_3/Shape:output:0?model_3/sequential_2/center_crop_3/strided_slice/stack:output:0Amodel_3/sequential_2/center_crop_3/strided_slice/stack_1:output:0Amodel_3/sequential_2/center_crop_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
(model_3/sequential_2/center_crop_3/sub/yConst*
_output_shapes
: *
dtype0*
value
B :??
&model_3/sequential_2/center_crop_3/subSub9model_3/sequential_2/center_crop_3/strided_slice:output:01model_3/sequential_2/center_crop_3/sub/y:output:0*
T0*
_output_shapes
: ?
8model_3/sequential_2/center_crop_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
:model_3/sequential_2/center_crop_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
:model_3/sequential_2/center_crop_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2model_3/sequential_2/center_crop_3/strided_slice_1StridedSlice1model_3/sequential_2/center_crop_3/Shape:output:0Amodel_3/sequential_2/center_crop_3/strided_slice_1/stack:output:0Cmodel_3/sequential_2/center_crop_3/strided_slice_1/stack_1:output:0Cmodel_3/sequential_2/center_crop_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
*model_3/sequential_2/center_crop_3/sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
(model_3/sequential_2/center_crop_3/sub_1Sub;model_3/sequential_2/center_crop_3/strided_slice_1:output:03model_3/sequential_2/center_crop_3/sub_1/y:output:0*
T0*
_output_shapes
: s
1model_3/sequential_2/center_crop_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
/model_3/sequential_2/center_crop_3/GreaterEqualGreaterEqual*model_3/sequential_2/center_crop_3/sub:z:0:model_3/sequential_2/center_crop_3/GreaterEqual/y:output:0*
T0*
_output_shapes
: u
3model_3/sequential_2/center_crop_3/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
1model_3/sequential_2/center_crop_3/GreaterEqual_1GreaterEqual,model_3/sequential_2/center_crop_3/sub_1:z:0<model_3/sequential_2/center_crop_3/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
.model_3/sequential_2/center_crop_3/Rank/packedPack3model_3/sequential_2/center_crop_3/GreaterEqual:z:05model_3/sequential_2/center_crop_3/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:i
'model_3/sequential_2/center_crop_3/RankConst*
_output_shapes
: *
dtype0*
value	B :p
.model_3/sequential_2/center_crop_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : p
.model_3/sequential_2/center_crop_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
(model_3/sequential_2/center_crop_3/rangeRange7model_3/sequential_2/center_crop_3/range/start:output:00model_3/sequential_2/center_crop_3/Rank:output:07model_3/sequential_2/center_crop_3/range/delta:output:0*
_output_shapes
:?
,model_3/sequential_2/center_crop_3/All/inputPack3model_3/sequential_2/center_crop_3/GreaterEqual:z:05model_3/sequential_2/center_crop_3/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:?
&model_3/sequential_2/center_crop_3/AllAll5model_3/sequential_2/center_crop_3/All/input:output:01model_3/sequential_2/center_crop_3/range:output:0*
_output_shapes
: ?
'model_3/sequential_2/center_crop_3/condIf/model_3/sequential_2/center_crop_3/All:output:0*model_3/sequential_2/center_crop_3/sub:z:0,model_3/sequential_2/center_crop_3/sub_1:z:0)model_3/tf.nn.bias_add_3/BiasAdd:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *F
else_branch7R5
3model_3_sequential_2_center_crop_3_cond_false_11967*0
output_shapes
:???????????*E
then_branch6R4
2model_3_sequential_2_center_crop_3_cond_true_11966?
0model_3/sequential_2/center_crop_3/cond/IdentityIdentity0model_3/sequential_2/center_crop_3/cond:output:0*
T0*1
_output_shapes
:????????????
(model_3/sequential_2/random_crop_2/ShapeShape9model_3/sequential_2/center_crop_3/cond/Identity:output:0*
T0*
_output_shapes
:?
6model_3/sequential_2/random_crop_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
8model_3/sequential_2/random_crop_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
8model_3/sequential_2/random_crop_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0model_3/sequential_2/random_crop_2/strided_sliceStridedSlice1model_3/sequential_2/random_crop_2/Shape:output:0?model_3/sequential_2/random_crop_2/strided_slice/stack:output:0Amodel_3/sequential_2/random_crop_2/strided_slice/stack_1:output:0Amodel_3/sequential_2/random_crop_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
(model_3/sequential_2/random_crop_2/sub/yConst*
_output_shapes
: *
dtype0*
value
B :??
&model_3/sequential_2/random_crop_2/subSub9model_3/sequential_2/random_crop_2/strided_slice:output:01model_3/sequential_2/random_crop_2/sub/y:output:0*
T0*
_output_shapes
: ?
8model_3/sequential_2/random_crop_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
:model_3/sequential_2/random_crop_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
:model_3/sequential_2/random_crop_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
2model_3/sequential_2/random_crop_2/strided_slice_1StridedSlice1model_3/sequential_2/random_crop_2/Shape:output:0Amodel_3/sequential_2/random_crop_2/strided_slice_1/stack:output:0Cmodel_3/sequential_2/random_crop_2/strided_slice_1/stack_1:output:0Cmodel_3/sequential_2/random_crop_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
*model_3/sequential_2/random_crop_2/sub_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
(model_3/sequential_2/random_crop_2/sub_1Sub;model_3/sequential_2/random_crop_2/strided_slice_1:output:03model_3/sequential_2/random_crop_2/sub_1/y:output:0*
T0*
_output_shapes
: s
1model_3/sequential_2/random_crop_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
/model_3/sequential_2/random_crop_2/GreaterEqualGreaterEqual*model_3/sequential_2/random_crop_2/sub:z:0:model_3/sequential_2/random_crop_2/GreaterEqual/y:output:0*
T0*
_output_shapes
: u
3model_3/sequential_2/random_crop_2/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
1model_3/sequential_2/random_crop_2/GreaterEqual_1GreaterEqual,model_3/sequential_2/random_crop_2/sub_1:z:0<model_3/sequential_2/random_crop_2/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: r
0model_3/sequential_2/random_crop_2/Rank/packed/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
.model_3/sequential_2/random_crop_2/Rank/packedPack9model_3/sequential_2/random_crop_2/Rank/packed/0:output:03model_3/sequential_2/random_crop_2/GreaterEqual:z:05model_3/sequential_2/random_crop_2/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:i
'model_3/sequential_2/random_crop_2/RankConst*
_output_shapes
: *
dtype0*
value	B :p
.model_3/sequential_2/random_crop_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : p
.model_3/sequential_2/random_crop_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
(model_3/sequential_2/random_crop_2/rangeRange7model_3/sequential_2/random_crop_2/range/start:output:00model_3/sequential_2/random_crop_2/Rank:output:07model_3/sequential_2/random_crop_2/range/delta:output:0*
_output_shapes
:p
.model_3/sequential_2/random_crop_2/All/input/0Const*
_output_shapes
: *
dtype0
*
value	B
 Z ?
,model_3/sequential_2/random_crop_2/All/inputPack7model_3/sequential_2/random_crop_2/All/input/0:output:03model_3/sequential_2/random_crop_2/GreaterEqual:z:05model_3/sequential_2/random_crop_2/GreaterEqual_1:z:0*
N*
T0
*
_output_shapes
:?
&model_3/sequential_2/random_crop_2/AllAll5model_3/sequential_2/random_crop_2/All/input:output:01model_3/sequential_2/random_crop_2/range:output:0*
_output_shapes
: ?
'model_3/sequential_2/random_crop_2/condIf/model_3/sequential_2/random_crop_2/All:output:0/model_3_sequential_2_random_crop_2_cond_input_0*model_3/sequential_2/random_crop_2/sub:z:0,model_3/sequential_2/random_crop_2/sub_1:z:09model_3/sequential_2/center_crop_3/cond/Identity:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *F
else_branch7R5
3model_3_sequential_2_random_crop_2_cond_false_12102*0
output_shapes
:???????????*E
then_branch6R4
2model_3_sequential_2_random_crop_2_cond_true_12101?
0model_3/sequential_2/random_crop_2/cond/IdentityIdentity0model_3/sequential_2/random_crop_2/cond:output:0*
T0*1
_output_shapes
:????????????
0model_3/vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
!model_3/vgg16/block1_conv1/Conv2DConv2D9model_3/sequential_2/random_crop_2/cond/Identity:output:08model_3/vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
1model_3/vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"model_3/vgg16/block1_conv1/BiasAddBiasAdd*model_3/vgg16/block1_conv1/Conv2D:output:09model_3/vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
model_3/vgg16/block1_conv1/ReluRelu+model_3/vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
0model_3/vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
!model_3/vgg16/block1_conv2/Conv2DConv2D-model_3/vgg16/block1_conv1/Relu:activations:08model_3/vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
1model_3/vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"model_3/vgg16/block1_conv2/BiasAddBiasAdd*model_3/vgg16/block1_conv2/Conv2D:output:09model_3/vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@?
model_3/vgg16/block1_conv2/ReluRelu+model_3/vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
!model_3/vgg16/block1_pool/MaxPoolMaxPool-model_3/vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????pp@*
ksize
*
paddingVALID*
strides
?
0model_3/vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
!model_3/vgg16/block2_conv1/Conv2DConv2D*model_3/vgg16/block1_pool/MaxPool:output:08model_3/vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
?
1model_3/vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_3/vgg16/block2_conv1/BiasAddBiasAdd*model_3/vgg16/block2_conv1/Conv2D:output:09model_3/vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp??
model_3/vgg16/block2_conv1/ReluRelu+model_3/vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp??
0model_3/vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!model_3/vgg16/block2_conv2/Conv2DConv2D-model_3/vgg16/block2_conv1/Relu:activations:08model_3/vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp?*
paddingSAME*
strides
?
1model_3/vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_3/vgg16/block2_conv2/BiasAddBiasAdd*model_3/vgg16/block2_conv2/Conv2D:output:09model_3/vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????pp??
model_3/vgg16/block2_conv2/ReluRelu+model_3/vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????pp??
!model_3/vgg16/block2_pool/MaxPoolMaxPool-model_3/vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:?????????88?*
ksize
*
paddingVALID*
strides
?
0model_3/vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!model_3/vgg16/block3_conv1/Conv2DConv2D*model_3/vgg16/block2_pool/MaxPool:output:08model_3/vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
1model_3/vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_3/vgg16/block3_conv1/BiasAddBiasAdd*model_3/vgg16/block3_conv1/Conv2D:output:09model_3/vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88??
model_3/vgg16/block3_conv1/ReluRelu+model_3/vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
0model_3/vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!model_3/vgg16/block3_conv2/Conv2DConv2D-model_3/vgg16/block3_conv1/Relu:activations:08model_3/vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
1model_3/vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_3/vgg16/block3_conv2/BiasAddBiasAdd*model_3/vgg16/block3_conv2/Conv2D:output:09model_3/vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88??
model_3/vgg16/block3_conv2/ReluRelu+model_3/vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
0model_3/vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!model_3/vgg16/block3_conv3/Conv2DConv2D-model_3/vgg16/block3_conv2/Relu:activations:08model_3/vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
?
1model_3/vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_3/vgg16/block3_conv3/BiasAddBiasAdd*model_3/vgg16/block3_conv3/Conv2D:output:09model_3/vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88??
model_3/vgg16/block3_conv3/ReluRelu+model_3/vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????88??
!model_3/vgg16/block3_pool/MaxPoolMaxPool-model_3/vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
0model_3/vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!model_3/vgg16/block4_conv1/Conv2DConv2D*model_3/vgg16/block3_pool/MaxPool:output:08model_3/vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
1model_3/vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_3/vgg16/block4_conv1/BiasAddBiasAdd*model_3/vgg16/block4_conv1/Conv2D:output:09model_3/vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
model_3/vgg16/block4_conv1/ReluRelu+model_3/vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
0model_3/vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!model_3/vgg16/block4_conv2/Conv2DConv2D-model_3/vgg16/block4_conv1/Relu:activations:08model_3/vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
1model_3/vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_3/vgg16/block4_conv2/BiasAddBiasAdd*model_3/vgg16/block4_conv2/Conv2D:output:09model_3/vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
model_3/vgg16/block4_conv2/ReluRelu+model_3/vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
0model_3/vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!model_3/vgg16/block4_conv3/Conv2DConv2D-model_3/vgg16/block4_conv2/Relu:activations:08model_3/vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
1model_3/vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_3/vgg16/block4_conv3/BiasAddBiasAdd*model_3/vgg16/block4_conv3/Conv2D:output:09model_3/vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
model_3/vgg16/block4_conv3/ReluRelu+model_3/vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
!model_3/vgg16/block4_pool/MaxPoolMaxPool-model_3/vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
0model_3/vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!model_3/vgg16/block5_conv1/Conv2DConv2D*model_3/vgg16/block4_pool/MaxPool:output:08model_3/vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
1model_3/vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_3/vgg16/block5_conv1/BiasAddBiasAdd*model_3/vgg16/block5_conv1/Conv2D:output:09model_3/vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
model_3/vgg16/block5_conv1/ReluRelu+model_3/vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
0model_3/vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!model_3/vgg16/block5_conv2/Conv2DConv2D-model_3/vgg16/block5_conv1/Relu:activations:08model_3/vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
1model_3/vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_3/vgg16/block5_conv2/BiasAddBiasAdd*model_3/vgg16/block5_conv2/Conv2D:output:09model_3/vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
model_3/vgg16/block5_conv2/ReluRelu+model_3/vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
0model_3/vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp9model_3_vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
!model_3/vgg16/block5_conv3/Conv2DConv2D-model_3/vgg16/block5_conv2/Relu:activations:08model_3/vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
1model_3/vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp:model_3_vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_3/vgg16/block5_conv3/BiasAddBiasAdd*model_3/vgg16/block5_conv3/Conv2D:output:09model_3/vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
model_3/vgg16/block5_conv3/ReluRelu+model_3/vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
!model_3/vgg16/block5_pool/MaxPoolMaxPool-model_3/vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
9model_3/global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
'model_3/global_average_pooling2d_3/MeanMean*model_3/vgg16/block5_pool/MaxPool:output:0Bmodel_3/global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
%model_3/dense_3/MatMul/ReadVariableOpReadVariableOp.model_3_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_3/dense_3/MatMulMatMul0model_3/global_average_pooling2d_3/Mean:output:0-model_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_3_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_3/dense_3/BiasAddBiasAdd model_3/dense_3/MatMul:product:0.model_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????o
IdentityIdentity model_3/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp'^model_3/dense_3/BiasAdd/ReadVariableOp&^model_3/dense_3/MatMul/ReadVariableOp(^model_3/sequential_2/center_crop_3/cond(^model_3/sequential_2/random_crop_2/cond2^model_3/vgg16/block1_conv1/BiasAdd/ReadVariableOp1^model_3/vgg16/block1_conv1/Conv2D/ReadVariableOp2^model_3/vgg16/block1_conv2/BiasAdd/ReadVariableOp1^model_3/vgg16/block1_conv2/Conv2D/ReadVariableOp2^model_3/vgg16/block2_conv1/BiasAdd/ReadVariableOp1^model_3/vgg16/block2_conv1/Conv2D/ReadVariableOp2^model_3/vgg16/block2_conv2/BiasAdd/ReadVariableOp1^model_3/vgg16/block2_conv2/Conv2D/ReadVariableOp2^model_3/vgg16/block3_conv1/BiasAdd/ReadVariableOp1^model_3/vgg16/block3_conv1/Conv2D/ReadVariableOp2^model_3/vgg16/block3_conv2/BiasAdd/ReadVariableOp1^model_3/vgg16/block3_conv2/Conv2D/ReadVariableOp2^model_3/vgg16/block3_conv3/BiasAdd/ReadVariableOp1^model_3/vgg16/block3_conv3/Conv2D/ReadVariableOp2^model_3/vgg16/block4_conv1/BiasAdd/ReadVariableOp1^model_3/vgg16/block4_conv1/Conv2D/ReadVariableOp2^model_3/vgg16/block4_conv2/BiasAdd/ReadVariableOp1^model_3/vgg16/block4_conv2/Conv2D/ReadVariableOp2^model_3/vgg16/block4_conv3/BiasAdd/ReadVariableOp1^model_3/vgg16/block4_conv3/Conv2D/ReadVariableOp2^model_3/vgg16/block5_conv1/BiasAdd/ReadVariableOp1^model_3/vgg16/block5_conv1/Conv2D/ReadVariableOp2^model_3/vgg16/block5_conv2/BiasAdd/ReadVariableOp1^model_3/vgg16/block5_conv2/Conv2D/ReadVariableOp2^model_3/vgg16/block5_conv3/BiasAdd/ReadVariableOp1^model_3/vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&model_3/dense_3/BiasAdd/ReadVariableOp&model_3/dense_3/BiasAdd/ReadVariableOp2N
%model_3/dense_3/MatMul/ReadVariableOp%model_3/dense_3/MatMul/ReadVariableOp2R
'model_3/sequential_2/center_crop_3/cond'model_3/sequential_2/center_crop_3/cond2R
'model_3/sequential_2/random_crop_2/cond'model_3/sequential_2/random_crop_2/cond2f
1model_3/vgg16/block1_conv1/BiasAdd/ReadVariableOp1model_3/vgg16/block1_conv1/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block1_conv1/Conv2D/ReadVariableOp0model_3/vgg16/block1_conv1/Conv2D/ReadVariableOp2f
1model_3/vgg16/block1_conv2/BiasAdd/ReadVariableOp1model_3/vgg16/block1_conv2/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block1_conv2/Conv2D/ReadVariableOp0model_3/vgg16/block1_conv2/Conv2D/ReadVariableOp2f
1model_3/vgg16/block2_conv1/BiasAdd/ReadVariableOp1model_3/vgg16/block2_conv1/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block2_conv1/Conv2D/ReadVariableOp0model_3/vgg16/block2_conv1/Conv2D/ReadVariableOp2f
1model_3/vgg16/block2_conv2/BiasAdd/ReadVariableOp1model_3/vgg16/block2_conv2/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block2_conv2/Conv2D/ReadVariableOp0model_3/vgg16/block2_conv2/Conv2D/ReadVariableOp2f
1model_3/vgg16/block3_conv1/BiasAdd/ReadVariableOp1model_3/vgg16/block3_conv1/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block3_conv1/Conv2D/ReadVariableOp0model_3/vgg16/block3_conv1/Conv2D/ReadVariableOp2f
1model_3/vgg16/block3_conv2/BiasAdd/ReadVariableOp1model_3/vgg16/block3_conv2/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block3_conv2/Conv2D/ReadVariableOp0model_3/vgg16/block3_conv2/Conv2D/ReadVariableOp2f
1model_3/vgg16/block3_conv3/BiasAdd/ReadVariableOp1model_3/vgg16/block3_conv3/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block3_conv3/Conv2D/ReadVariableOp0model_3/vgg16/block3_conv3/Conv2D/ReadVariableOp2f
1model_3/vgg16/block4_conv1/BiasAdd/ReadVariableOp1model_3/vgg16/block4_conv1/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block4_conv1/Conv2D/ReadVariableOp0model_3/vgg16/block4_conv1/Conv2D/ReadVariableOp2f
1model_3/vgg16/block4_conv2/BiasAdd/ReadVariableOp1model_3/vgg16/block4_conv2/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block4_conv2/Conv2D/ReadVariableOp0model_3/vgg16/block4_conv2/Conv2D/ReadVariableOp2f
1model_3/vgg16/block4_conv3/BiasAdd/ReadVariableOp1model_3/vgg16/block4_conv3/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block4_conv3/Conv2D/ReadVariableOp0model_3/vgg16/block4_conv3/Conv2D/ReadVariableOp2f
1model_3/vgg16/block5_conv1/BiasAdd/ReadVariableOp1model_3/vgg16/block5_conv1/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block5_conv1/Conv2D/ReadVariableOp0model_3/vgg16/block5_conv1/Conv2D/ReadVariableOp2f
1model_3/vgg16/block5_conv2/BiasAdd/ReadVariableOp1model_3/vgg16/block5_conv2/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block5_conv2/Conv2D/ReadVariableOp0model_3/vgg16/block5_conv2/Conv2D/ReadVariableOp2f
1model_3/vgg16/block5_conv3/BiasAdd/ReadVariableOp1model_3/vgg16/block5_conv3/BiasAdd/ReadVariableOp2d
0model_3/vgg16/block5_conv3/Conv2D/ReadVariableOp0model_3/vgg16/block5_conv3/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_8: 

_output_shapes
:
?
V
:__inference_global_average_pooling2d_3_layer_call_fn_16461

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_13849i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?'
?	
B__inference_model_3_layer_call_and_return_conditional_losses_14153

inputs!
tf_nn_bias_add_3_biasadd_bias 
sequential_2_14088:	 
sequential_2_14090:	%
vgg16_14093:@
vgg16_14095:@%
vgg16_14097:@@
vgg16_14099:@&
vgg16_14101:@?
vgg16_14103:	?'
vgg16_14105:??
vgg16_14107:	?'
vgg16_14109:??
vgg16_14111:	?'
vgg16_14113:??
vgg16_14115:	?'
vgg16_14117:??
vgg16_14119:	?'
vgg16_14121:??
vgg16_14123:	?'
vgg16_14125:??
vgg16_14127:	?'
vgg16_14129:??
vgg16_14131:	?'
vgg16_14133:??
vgg16_14135:	?'
vgg16_14137:??
vgg16_14139:	?'
vgg16_14141:??
vgg16_14143:	? 
dense_3_14147:	?
dense_3_14149:
identity??dense_3/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?vgg16/StatefulPartitionedCall
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"   ?????
(tf.__operators__.getitem_3/strided_sliceStridedSliceinputs7tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask*
ellipsis_mask*
end_mask?
tf.nn.bias_add_3/BiasAddBiasAdd1tf.__operators__.getitem_3/strided_slice:output:0tf_nn_bias_add_3_biasadd_bias*
T0*1
_output_shapes
:????????????
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall!tf.nn.bias_add_3/BiasAdd:output:0sequential_2_14088sequential_2_14090*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_12931?
vgg16/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0vgg16_14093vgg16_14095vgg16_14097vgg16_14099vgg16_14101vgg16_14103vgg16_14105vgg16_14107vgg16_14109vgg16_14111vgg16_14113vgg16_14115vgg16_14117vgg16_14119vgg16_14121vgg16_14123vgg16_14125vgg16_14127vgg16_14129vgg16_14131vgg16_14133vgg16_14135vgg16_14137vgg16_14139vgg16_14141vgg16_14143*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_13261?
*global_average_pooling2d_3/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_13849?
dense_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0dense_3_14147dense_3_14149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_13932w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_3/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*r
_input_shapesa
_:???????????:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs: 

_output_shapes
:
?
?
G__inference_block3_conv1_layer_call_and_return_conditional_losses_17151

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????88?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????88?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????88?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????88?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????88?
 
_user_specified_nameinputs
?Y
?
center_crop_3_cond_true_155345
1center_crop_3_cond_truediv_cast_center_crop_3_sub9
5center_crop_3_cond_truediv_1_cast_center_crop_3_sub_18
4center_crop_3_cond_crop_to_bounding_box_shape_inputs
center_crop_3_cond_identity??5center_crop_3/cond/crop_to_bounding_box/Assert/Assert?7center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert?7center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert?7center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert^
center_crop_3/cond/truediv/yConst*
_output_shapes
: *
dtype0*
value	B :?
center_crop_3/cond/truediv/CastCast1center_crop_3_cond_truediv_cast_center_crop_3_sub*

DstT0*

SrcT0*
_output_shapes
: ?
!center_crop_3/cond/truediv/Cast_1Cast%center_crop_3/cond/truediv/y:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
center_crop_3/cond/truedivRealDiv#center_crop_3/cond/truediv/Cast:y:0%center_crop_3/cond/truediv/Cast_1:y:0*
T0*
_output_shapes
: o
center_crop_3/cond/CastCastcenter_crop_3/cond/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
: `
center_crop_3/cond/truediv_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
!center_crop_3/cond/truediv_1/CastCast5center_crop_3_cond_truediv_1_cast_center_crop_3_sub_1*

DstT0*

SrcT0*
_output_shapes
: ?
#center_crop_3/cond/truediv_1/Cast_1Cast'center_crop_3/cond/truediv_1/y:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
center_crop_3/cond/truediv_1RealDiv%center_crop_3/cond/truediv_1/Cast:y:0'center_crop_3/cond/truediv_1/Cast_1:y:0*
T0*
_output_shapes
: s
center_crop_3/cond/Cast_1Cast center_crop_3/cond/truediv_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ?
-center_crop_3/cond/crop_to_bounding_box/ShapeShape4center_crop_3_cond_crop_to_bounding_box_shape_inputs*
T0*
_output_shapes
:?
/center_crop_3/cond/crop_to_bounding_box/unstackUnpack6center_crop_3/cond/crop_to_bounding_box/Shape:output:0*
T0*
_output_shapes

: : : : *	
numx
6center_crop_3/cond/crop_to_bounding_box/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ?
4center_crop_3/cond/crop_to_bounding_box/GreaterEqualGreaterEqualcenter_crop_3/cond/Cast_1:y:0?center_crop_3/cond/crop_to_bounding_box/GreaterEqual/y:output:0*
T0*
_output_shapes
: ?
4center_crop_3/cond/crop_to_bounding_box/Assert/ConstConst*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
<center_crop_3/cond/crop_to_bounding_box/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*+
value"B  Boffset_width must be >= 0.?
5center_crop_3/cond/crop_to_bounding_box/Assert/AssertAssert8center_crop_3/cond/crop_to_bounding_box/GreaterEqual:z:0Ecenter_crop_3/cond/crop_to_bounding_box/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 z
8center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1/yConst*
_output_shapes
: *
dtype0*
value	B : ?
6center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1GreaterEqualcenter_crop_3/cond/Cast:y:0Acenter_crop_3/cond/crop_to_bounding_box/GreaterEqual_1/y:output:0*
T0*
_output_shapes
: ?
6center_crop_3/cond/crop_to_bounding_box/Assert_1/ConstConst*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
>center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert/data_0Const*
_output_shapes
: *
dtype0*,
value#B! Boffset_height must be >= 0.?
7center_crop_3/cond/crop_to_bounding_box/Assert_1/AssertAssert:center_crop_3/cond/crop_to_bounding_box/GreaterEqual_1:z:0Gcenter_crop_3/cond/crop_to_bounding_box/Assert_1/Assert/data_0:output:06^center_crop_3/cond/crop_to_bounding_box/Assert/Assert*

T
2*
_output_shapes
 p
-center_crop_3/cond/crop_to_bounding_box/add/xConst*
_output_shapes
: *
dtype0*
value
B :??
+center_crop_3/cond/crop_to_bounding_box/addAddV26center_crop_3/cond/crop_to_bounding_box/add/x:output:0center_crop_3/cond/Cast_1:y:0*
T0*
_output_shapes
: v
3center_crop_3/cond/crop_to_bounding_box/LessEqual/yConst*
_output_shapes
: *
dtype0*
value
B :??
1center_crop_3/cond/crop_to_bounding_box/LessEqual	LessEqual/center_crop_3/cond/crop_to_bounding_box/add:z:0<center_crop_3/cond/crop_to_bounding_box/LessEqual/y:output:0*
T0*
_output_shapes
: ?
6center_crop_3/cond/crop_to_bounding_box/Assert_2/ConstConst*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
>center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert/data_0Const*
_output_shapes
: *
dtype0*2
value)B' B!width must be >= target + offset.?
7center_crop_3/cond/crop_to_bounding_box/Assert_2/AssertAssert5center_crop_3/cond/crop_to_bounding_box/LessEqual:z:0Gcenter_crop_3/cond/crop_to_bounding_box/Assert_2/Assert/data_0:output:08^center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert*

T
2*
_output_shapes
 r
/center_crop_3/cond/crop_to_bounding_box/add_1/xConst*
_output_shapes
: *
dtype0*
value
B :??
-center_crop_3/cond/crop_to_bounding_box/add_1AddV28center_crop_3/cond/crop_to_bounding_box/add_1/x:output:0center_crop_3/cond/Cast:y:0*
T0*
_output_shapes
: x
5center_crop_3/cond/crop_to_bounding_box/LessEqual_1/yConst*
_output_shapes
: *
dtype0*
value
B :??
3center_crop_3/cond/crop_to_bounding_box/LessEqual_1	LessEqual1center_crop_3/cond/crop_to_bounding_box/add_1:z:0>center_crop_3/cond/crop_to_bounding_box/LessEqual_1/y:output:0*
T0*
_output_shapes
: ?
6center_crop_3/cond/crop_to_bounding_box/Assert_3/ConstConst*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
>center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert/data_0Const*
_output_shapes
: *
dtype0*3
value*B( B"height must be >= target + offset.?
7center_crop_3/cond/crop_to_bounding_box/Assert_3/AssertAssert7center_crop_3/cond/crop_to_bounding_box/LessEqual_1:z:0Gcenter_crop_3/cond/crop_to_bounding_box/Assert_3/Assert/data_0:output:08^center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert*

T
2*
_output_shapes
 ?
:center_crop_3/cond/crop_to_bounding_box/control_dependencyIdentity4center_crop_3_cond_crop_to_bounding_box_shape_inputs6^center_crop_3/cond/crop_to_bounding_box/Assert/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert*
T0*
_class
loc:@inputs*1
_output_shapes
:???????????q
/center_crop_3/cond/crop_to_bounding_box/stack/0Const*
_output_shapes
: *
dtype0*
value	B : q
/center_crop_3/cond/crop_to_bounding_box/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
-center_crop_3/cond/crop_to_bounding_box/stackPack8center_crop_3/cond/crop_to_bounding_box/stack/0:output:0center_crop_3/cond/Cast:y:0center_crop_3/cond/Cast_1:y:08center_crop_3/cond/crop_to_bounding_box/stack/3:output:0*
N*
T0*
_output_shapes
:?
/center_crop_3/cond/crop_to_bounding_box/Shape_1ShapeCcenter_crop_3/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
;center_crop_3/cond/crop_to_bounding_box/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=center_crop_3/cond/crop_to_bounding_box/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5center_crop_3/cond/crop_to_bounding_box/strided_sliceStridedSlice8center_crop_3/cond/crop_to_bounding_box/Shape_1:output:0Dcenter_crop_3/cond/crop_to_bounding_box/strided_slice/stack:output:0Fcenter_crop_3/cond/crop_to_bounding_box/strided_slice/stack_1:output:0Fcenter_crop_3/cond/crop_to_bounding_box/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
/center_crop_3/cond/crop_to_bounding_box/Shape_2ShapeCcenter_crop_3/cond/crop_to_bounding_box/control_dependency:output:0*
T0*
_output_shapes
:?
=center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
?center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?center_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7center_crop_3/cond/crop_to_bounding_box/strided_slice_1StridedSlice8center_crop_3/cond/crop_to_bounding_box/Shape_2:output:0Fcenter_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack:output:0Hcenter_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_1:output:0Hcenter_crop_3/cond/crop_to_bounding_box/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
1center_crop_3/cond/crop_to_bounding_box/stack_1/1Const*
_output_shapes
: *
dtype0*
value
B :?t
1center_crop_3/cond/crop_to_bounding_box/stack_1/2Const*
_output_shapes
: *
dtype0*
value
B :??
/center_crop_3/cond/crop_to_bounding_box/stack_1Pack>center_crop_3/cond/crop_to_bounding_box/strided_slice:output:0:center_crop_3/cond/crop_to_bounding_box/stack_1/1:output:0:center_crop_3/cond/crop_to_bounding_box/stack_1/2:output:0@center_crop_3/cond/crop_to_bounding_box/strided_slice_1:output:0*
N*
T0*
_output_shapes
:?
-center_crop_3/cond/crop_to_bounding_box/SliceSliceCcenter_crop_3/cond/crop_to_bounding_box/control_dependency:output:06center_crop_3/cond/crop_to_bounding_box/stack:output:08center_crop_3/cond/crop_to_bounding_box/stack_1:output:0*
Index0*
T0*1
_output_shapes
:????????????
center_crop_3/cond/IdentityIdentity6center_crop_3/cond/crop_to_bounding_box/Slice:output:0^center_crop_3/cond/NoOp*
T0*1
_output_shapes
:????????????
center_crop_3/cond/NoOpNoOp6^center_crop_3/cond/crop_to_bounding_box/Assert/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert8^center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert*"
_acd_function_control_output(*
_output_shapes
 "C
center_crop_3_cond_identity$center_crop_3/cond/Identity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!: : :???????????2n
5center_crop_3/cond/crop_to_bounding_box/Assert/Assert5center_crop_3/cond/crop_to_bounding_box/Assert/Assert2r
7center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert7center_crop_3/cond/crop_to_bounding_box/Assert_1/Assert2r
7center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert7center_crop_3/cond/crop_to_bounding_box/Assert_2/Assert2r
7center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert7center_crop_3/cond/crop_to_bounding_box/Assert_3/Assert: 

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:???????????"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_8:
serving_default_input_8:0???????????;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer-5
layer_with_weights-1
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
?
layer-0
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
 layer_with_weights-2
 layer-4
!layer_with_weights-3
!layer-5
"layer-6
#layer_with_weights-4
#layer-7
$layer_with_weights-5
$layer-8
%layer_with_weights-6
%layer-9
&layer-10
'layer_with_weights-7
'layer-11
(layer_with_weights-8
(layer-12
)layer_with_weights-9
)layer-13
*layer-14
+layer_with_weights-10
+layer-15
,layer_with_weights-11
,layer-16
-layer_with_weights-12
-layer-17
.layer-18
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_network
?
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
?

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Citer
	Ddecay
Elearning_rate
Fmomentum
Grho
;rms?
<rms?
\rms?
]rms?
^rms?
_rms?
`rms?
arms?"
	optimizer
?
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19
\20
]21
^22
_23
`24
a25
;26
<27"
trackable_list_wrapper
X
\0
]1
^2
_3
`4
a5
;6
<7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_model_3_layer_call_fn_14002
'__inference_model_3_layer_call_fn_14502
'__inference_model_3_layer_call_fn_14569
'__inference_model_3_layer_call_fn_14285?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_3_layer_call_and_return_conditional_losses_14978
B__inference_model_3_layer_call_and_return_conditional_losses_15424
B__inference_model_3_layer_call_and_return_conditional_losses_14357
B__inference_model_3_layer_call_and_return_conditional_losses_14431?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_12342input_8"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
gserving_default"
signature_map
"
_generic_user_object
"
_generic_user_object
?
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
?
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r_random_generator
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
?
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y_random_generator
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_sequential_2_layer_call_fn_12670
,__inference_sequential_2_layer_call_fn_15498
,__inference_sequential_2_layer_call_fn_15507
,__inference_sequential_2_layer_call_fn_12947?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15806
G__inference_sequential_2_layer_call_and_return_conditional_losses_16142
G__inference_sequential_2_layer_call_and_return_conditional_losses_12956
G__inference_sequential_2_layer_call_and_return_conditional_losses_12967?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_tf_keras_input_layer
?

Hkernel
Ibias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Jkernel
Kbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Lkernel
Mbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Nkernel
Obias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Pkernel
Qbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Rkernel
Sbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Tkernel
Ubias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Vkernel
Wbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Xkernel
Ybias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Zkernel
[bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

\kernel
]bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

^kernel
_bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

`kernel
abias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19
\20
]21
^22
_23
`24
a25"
trackable_list_wrapper
J
\0
]1
^2
_3
`4
a5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
?2?
%__inference_vgg16_layer_call_fn_13316
%__inference_vgg16_layer_call_fn_16199
%__inference_vgg16_layer_call_fn_16256
%__inference_vgg16_layer_call_fn_13691?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_vgg16_layer_call_and_return_conditional_losses_16356
@__inference_vgg16_layer_call_and_return_conditional_losses_16456
@__inference_vgg16_layer_call_and_return_conditional_losses_13765
@__inference_vgg16_layer_call_and_return_conditional_losses_13839?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
?2?
:__inference_global_average_pooling2d_3_layer_call_fn_16461?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_16467?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
!:	?2dense_3/kernel
:2dense_3/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_dense_3_layer_call_fn_16476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_16486?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@?2block2_conv1/kernel
 :?2block2_conv1/bias
/:-??2block2_conv2/kernel
 :?2block2_conv2/bias
/:-??2block3_conv1/kernel
 :?2block3_conv1/bias
/:-??2block3_conv2/kernel
 :?2block3_conv2/bias
/:-??2block3_conv3/kernel
 :?2block3_conv3/bias
/:-??2block4_conv1/kernel
 :?2block4_conv1/bias
/:-??2block4_conv2/kernel
 :?2block4_conv2/bias
/:-??2block4_conv3/kernel
 :?2block4_conv3/bias
/:-??2block5_conv1/kernel
 :?2block5_conv1/bias
/:-??2block5_conv2/kernel
 :?2block5_conv2/bias
/:-??2block5_conv3/kernel
 :?2block5_conv3/bias
?
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
#__inference_signature_wrapper_15491input_8"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_center_crop_3_layer_call_fn_16491?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_center_crop_3_layer_call_and_return_conditional_losses_16628?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
n	variables
otrainable_variables
pregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
/
?
_generator"
_generic_user_object
?2?
-__inference_random_crop_2_layer_call_fn_16635
-__inference_random_crop_2_layer_call_fn_16642?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_random_crop_2_layer_call_and_return_conditional_losses_16808
H__inference_random_crop_2_layer_call_and_return_conditional_losses_16974?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
/
?
_generator"
_generic_user_object
?2?
1__inference_random_contrast_2_layer_call_fn_16979
1__inference_random_contrast_2_layer_call_fn_16986?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_16990
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_17031?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block1_conv1_layer_call_fn_17040?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block1_conv1_layer_call_and_return_conditional_losses_17051?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block1_conv2_layer_call_fn_17060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_17071?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_block1_pool_layer_call_fn_17076?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block1_pool_layer_call_and_return_conditional_losses_17081?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block2_conv1_layer_call_fn_17090?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block2_conv1_layer_call_and_return_conditional_losses_17101?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block2_conv2_layer_call_fn_17110?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block2_conv2_layer_call_and_return_conditional_losses_17121?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_block2_pool_layer_call_fn_17126?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block2_pool_layer_call_and_return_conditional_losses_17131?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block3_conv1_layer_call_fn_17140?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block3_conv1_layer_call_and_return_conditional_losses_17151?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block3_conv2_layer_call_fn_17160?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block3_conv2_layer_call_and_return_conditional_losses_17171?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block3_conv3_layer_call_fn_17180?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block3_conv3_layer_call_and_return_conditional_losses_17191?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_block3_pool_layer_call_fn_17196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_pool_layer_call_and_return_conditional_losses_17201?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block4_conv1_layer_call_fn_17210?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block4_conv1_layer_call_and_return_conditional_losses_17221?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block4_conv2_layer_call_fn_17230?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block4_conv2_layer_call_and_return_conditional_losses_17241?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block4_conv3_layer_call_fn_17250?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block4_conv3_layer_call_and_return_conditional_losses_17261?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_block4_pool_layer_call_fn_17266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_pool_layer_call_and_return_conditional_losses_17271?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block5_conv1_layer_call_fn_17280?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block5_conv1_layer_call_and_return_conditional_losses_17291?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block5_conv2_layer_call_fn_17300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block5_conv2_layer_call_and_return_conditional_losses_17311?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_block5_conv3_layer_call_fn_17320?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_block5_conv3_layer_call_and_return_conditional_losses_17331?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_block5_pool_layer_call_fn_17336?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block5_pool_layer_call_and_return_conditional_losses_17341?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
H0
I1
J2
K3
L4
M5
N6
O7
P8
Q9
R10
S11
T12
U13
V14
W15
X16
Y17
Z18
[19"
trackable_list_wrapper
?
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12
)13
*14
+15
,16
-17
.18"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
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
/
?
_state_var"
_generic_user_object
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
/
?
_state_var"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
J0
K1"
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
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
N0
O1"
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
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
T0
U1"
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
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Z0
[1"
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
':%	2random_crop_2/cond/StateVar
&:$	2random_contrast_2/StateVar
+:)	?2RMSprop/dense_3/kernel/rms
$:"2RMSprop/dense_3/bias/rms
9:7??2RMSprop/block5_conv1/kernel/rms
*:(?2RMSprop/block5_conv1/bias/rms
9:7??2RMSprop/block5_conv2/kernel/rms
*:(?2RMSprop/block5_conv2/bias/rms
9:7??2RMSprop/block5_conv3/kernel/rms
*:(?2RMSprop/block5_conv3/bias/rms
	J
Const?
 __inference__wrapped_model_12342? ??HIJKLMNOPQRSTUVWXYZ[\]^_`a;<:?7
0?-
+?(
input_8???????????
? "1?.
,
dense_3!?
dense_3??????????
G__inference_block1_conv1_layer_call_and_return_conditional_losses_17051pHI9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????@
? ?
,__inference_block1_conv1_layer_call_fn_17040cHI9?6
/?,
*?'
inputs???????????
? ""????????????@?
G__inference_block1_conv2_layer_call_and_return_conditional_losses_17071pJK9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
,__inference_block1_conv2_layer_call_fn_17060cJK9?6
/?,
*?'
inputs???????????@
? ""????????????@?
F__inference_block1_pool_layer_call_and_return_conditional_losses_17081?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block1_pool_layer_call_fn_17076?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_block2_conv1_layer_call_and_return_conditional_losses_17101mLM7?4
-?*
(?%
inputs?????????pp@
? ".?+
$?!
0?????????pp?
? ?
,__inference_block2_conv1_layer_call_fn_17090`LM7?4
-?*
(?%
inputs?????????pp@
? "!??????????pp??
G__inference_block2_conv2_layer_call_and_return_conditional_losses_17121nNO8?5
.?+
)?&
inputs?????????pp?
? ".?+
$?!
0?????????pp?
? ?
,__inference_block2_conv2_layer_call_fn_17110aNO8?5
.?+
)?&
inputs?????????pp?
? "!??????????pp??
F__inference_block2_pool_layer_call_and_return_conditional_losses_17131?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block2_pool_layer_call_fn_17126?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_block3_conv1_layer_call_and_return_conditional_losses_17151nPQ8?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
,__inference_block3_conv1_layer_call_fn_17140aPQ8?5
.?+
)?&
inputs?????????88?
? "!??????????88??
G__inference_block3_conv2_layer_call_and_return_conditional_losses_17171nRS8?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
,__inference_block3_conv2_layer_call_fn_17160aRS8?5
.?+
)?&
inputs?????????88?
? "!??????????88??
G__inference_block3_conv3_layer_call_and_return_conditional_losses_17191nTU8?5
.?+
)?&
inputs?????????88?
? ".?+
$?!
0?????????88?
? ?
,__inference_block3_conv3_layer_call_fn_17180aTU8?5
.?+
)?&
inputs?????????88?
? "!??????????88??
F__inference_block3_pool_layer_call_and_return_conditional_losses_17201?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block3_pool_layer_call_fn_17196?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_block4_conv1_layer_call_and_return_conditional_losses_17221nVW8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block4_conv1_layer_call_fn_17210aVW8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block4_conv2_layer_call_and_return_conditional_losses_17241nXY8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block4_conv2_layer_call_fn_17230aXY8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block4_conv3_layer_call_and_return_conditional_losses_17261nZ[8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block4_conv3_layer_call_fn_17250aZ[8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block4_pool_layer_call_and_return_conditional_losses_17271?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block4_pool_layer_call_fn_17266?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_block5_conv1_layer_call_and_return_conditional_losses_17291n\]8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block5_conv1_layer_call_fn_17280a\]8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block5_conv2_layer_call_and_return_conditional_losses_17311n^_8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block5_conv2_layer_call_fn_17300a^_8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_block5_conv3_layer_call_and_return_conditional_losses_17331n`a8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_block5_conv3_layer_call_fn_17320a`a8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block5_pool_layer_call_and_return_conditional_losses_17341?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
+__inference_block5_pool_layer_call_fn_17336?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_center_crop_3_layer_call_and_return_conditional_losses_16628l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
-__inference_center_crop_3_layer_call_fn_16491_9?6
/?,
*?'
inputs???????????
? ""?????????????
B__inference_dense_3_layer_call_and_return_conditional_losses_16486];<0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_3_layer_call_fn_16476P;<0?-
&?#
!?
inputs??????????
? "???????????
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_16467?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
:__inference_global_average_pooling2d_3_layer_call_fn_16461wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
B__inference_model_3_layer_call_and_return_conditional_losses_14357? ??HIJKLMNOPQRSTUVWXYZ[\]^_`a;<B??
8?5
+?(
input_8???????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_14431?"???HIJKLMNOPQRSTUVWXYZ[\]^_`a;<B??
8?5
+?(
input_8???????????
p

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_14978? ??HIJKLMNOPQRSTUVWXYZ[\]^_`a;<A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_3_layer_call_and_return_conditional_losses_15424?"???HIJKLMNOPQRSTUVWXYZ[\]^_`a;<A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
'__inference_model_3_layer_call_fn_14002? ??HIJKLMNOPQRSTUVWXYZ[\]^_`a;<B??
8?5
+?(
input_8???????????
p 

 
? "???????????
'__inference_model_3_layer_call_fn_14285?"???HIJKLMNOPQRSTUVWXYZ[\]^_`a;<B??
8?5
+?(
input_8???????????
p

 
? "???????????
'__inference_model_3_layer_call_fn_14502 ??HIJKLMNOPQRSTUVWXYZ[\]^_`a;<A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
'__inference_model_3_layer_call_fn_14569?"???HIJKLMNOPQRSTUVWXYZ[\]^_`a;<A?>
7?4
*?'
inputs???????????
p

 
? "???????????
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_16990p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
L__inference_random_contrast_2_layer_call_and_return_conditional_losses_17031t?=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
1__inference_random_contrast_2_layer_call_fn_16979c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
1__inference_random_contrast_2_layer_call_fn_16986g?=?:
3?0
*?'
inputs???????????
p
? ""?????????????
H__inference_random_crop_2_layer_call_and_return_conditional_losses_16808t?=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
H__inference_random_crop_2_layer_call_and_return_conditional_losses_16974t?=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
-__inference_random_crop_2_layer_call_fn_16635g?=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
-__inference_random_crop_2_layer_call_fn_16642g?=?:
3?0
*?'
inputs???????????
p
? ""?????????????
G__inference_sequential_2_layer_call_and_return_conditional_losses_12956??N?K
D?A
7?4
center_crop_3_input???????????
p 

 
? "/?,
%?"
0???????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_12967???N?K
D?A
7?4
center_crop_3_input???????????
p

 
? "/?,
%?"
0???????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_15806x?A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_16142z??A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
,__inference_sequential_2_layer_call_fn_12670x?N?K
D?A
7?4
center_crop_3_input???????????
p 

 
? ""?????????????
,__inference_sequential_2_layer_call_fn_12947z??N?K
D?A
7?4
center_crop_3_input???????????
p

 
? ""?????????????
,__inference_sequential_2_layer_call_fn_15498k?A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
,__inference_sequential_2_layer_call_fn_15507m??A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
#__inference_signature_wrapper_15491? ??HIJKLMNOPQRSTUVWXYZ[\]^_`a;<E?B
? 
;?8
6
input_8+?(
input_8???????????"1?.
,
dense_3!?
dense_3??????????
@__inference_vgg16_layer_call_and_return_conditional_losses_13765?HIJKLMNOPQRSTUVWXYZ[\]^_`aB??
8?5
+?(
input_7???????????
p 

 
? ".?+
$?!
0??????????
? ?
@__inference_vgg16_layer_call_and_return_conditional_losses_13839?HIJKLMNOPQRSTUVWXYZ[\]^_`aB??
8?5
+?(
input_7???????????
p

 
? ".?+
$?!
0??????????
? ?
@__inference_vgg16_layer_call_and_return_conditional_losses_16356?HIJKLMNOPQRSTUVWXYZ[\]^_`aA?>
7?4
*?'
inputs???????????
p 

 
? ".?+
$?!
0??????????
? ?
@__inference_vgg16_layer_call_and_return_conditional_losses_16456?HIJKLMNOPQRSTUVWXYZ[\]^_`aA?>
7?4
*?'
inputs???????????
p

 
? ".?+
$?!
0??????????
? ?
%__inference_vgg16_layer_call_fn_13316?HIJKLMNOPQRSTUVWXYZ[\]^_`aB??
8?5
+?(
input_7???????????
p 

 
? "!????????????
%__inference_vgg16_layer_call_fn_13691?HIJKLMNOPQRSTUVWXYZ[\]^_`aB??
8?5
+?(
input_7???????????
p

 
? "!????????????
%__inference_vgg16_layer_call_fn_16199?HIJKLMNOPQRSTUVWXYZ[\]^_`aA?>
7?4
*?'
inputs???????????
p 

 
? "!????????????
%__inference_vgg16_layer_call_fn_16256?HIJKLMNOPQRSTUVWXYZ[\]^_`aA?>
7?4
*?'
inputs???????????
p

 
? "!???????????