// EXPECTED_REWIRING [0 1 2 3 4 5 7 6 8 9 10 11 13 12 14 15]
// CURRENT_REWIRING [10 8 11 3 14 5 0 6 7 15 4 1 9 2 12 13]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.075736053375836*pi) q[1];
cz q[5], q[10];
rz(1.1384154740107841*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.942118687998284*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.9109514883197644*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.089477271439539*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.892929576718223*pi) q[6];
cz q[6], q[5];
rz(-3.002559241075489*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-2.087802470758894*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3844841619731474*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.2762476260936904*pi) q[9];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(2.8079966966946865*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.41543848027827707*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.2637533427426022*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.4597000595984486*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(0.24271325173163064*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.261599837637768*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.244269372863136*pi) q[1];
cz q[0], q[1];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(0.10344064106915161*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.053790182830899*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3844841619731472*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.436141354291*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[9], q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(2.761369489712264*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.9641888827222767*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.9438241621069082*pi) q[7];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
cz q[0], q[1];
rz(1.468484313433314*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.0125331012365915*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.5271142084493476*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.227269400656863*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.0112026142548045*pi) q[3];
cz q[3], q[2];
rz(2.2021291074121425*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.7437907301540037*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.0496921951537208*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.0699974014672198*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.3088502052605655*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.4162233093670955*pi) q[10];
cz q[10], q[9];
rz(1.3917871387349376*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(3.141592653589793*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(0.7740060436714727*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.2783532062637788*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.815728329663653*pi) q[6];
rz(0.7086722564037171*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.5284868738287399*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(-1.5707963267948966*pi) q[6];
rz(2.9960338991142255*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(-0.5888827565204153*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.075341084184929*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.7990453694137907*pi) q[7];
rz(2.6245865096257948*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.3844841619731463*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.2762476260936895*pi) q[8];
rz(1.3849592185268063*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.5724540101793967*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.1651934448621555*pi) q[5];
rz(-1.4867615583333942*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.482120250874175*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[11];
rz(-2.8038956306707945*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.7843290499389812*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.0779896335268964*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.381184772407101*pi) q[13];
cz q[13], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-1.9038013325814909*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.754423745298308*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(2.8039449583294878*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4416439879888072*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.608194203367244*pi) q[6];
rz(1.3775495739276882*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.642209555551563*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rx(-1.5707963267948966*pi) q[6];
rz(2.349044458710013*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[6], q[9];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rz(-0.15668060704271874*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.6596312975941372*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.5534299765886912*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(1.710482289701405*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.7843290499389812*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.077989633526896*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.8103884456122044*pi) q[15];
rz(-2.7311824310431803*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.0806210510449956*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.449400234830656*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.8051877820847997*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-1.1826746570991549*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.7370835733564736*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.41848320726723237*pi) q[6];
cz q[6], q[1];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-0.4892353831994205*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[1], q[6];
cz q[8], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-0.9845275665550752*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.6715533641657399*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.0609523713024975*pi) q[9];
rz(-1.8055042443797629*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.1088000422046747*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[9], q[10];
rx(-1.5707963267948966*pi) q[9];
rz(-1.5522563688335869*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[9], q[10];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[9], q[10];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687794*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.730367851897572*pi) q[8];
rz(-0.014496630227443364*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.1200934510635538*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.5771113931360865*pi) q[9];
cz q[8], q[9];
rz(2.2048178269432652*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.7633435501336161*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-1.189152600663699*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.2364127944007053*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(1.2743510324203786*pi) q[15];
cz q[15], q[14];
rz(-0.9469112563848494*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(3.141592653589793*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(3.141592653589793*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(1.0566141769368973*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(2.4547096471343477*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.1777663163788905*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.5780328071455365*pi) q[14];
rz(0.59368010174542*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.958108965734335*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.4741817804854853*pi) q[4];
rz(-2.214805987710589*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.2683907161509058*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[10], q[11];
cz q[9], q[14];
rz(-1.9811743791113248*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(0.2260688200448524*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.078628134225546*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.403817189325282*pi) q[7];
rz(1.0524175788931784*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.1923298298177034*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.9264339181161145*pi) q[8];
rz(-2.4317818302882857*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.5409303848861269*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[8], q[15];
rx(-1.5707963267948966*pi) q[8];
rz(0.15660861369803047*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[8], q[15];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[15];
cz q[8], q[15];
rz(2.07559513449817*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.3543636222221993*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.8315256931336119*pi) q[3];
rz(3.141592653589793*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[12];
rz(-0.213532723144084*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.0779896335268955*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.381184772407101*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[11];
rz(1.35726360365081*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.077989633526896*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-2.381184772407101*pi) q[12];
cz q[12], q[11];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(-2.1644764285403166*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.1834836878554581*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.667410873104308*pi) q[1];
rz(-1.7700105602691747*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.4762467683162614*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.9900411904080786*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.8700160176615566*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(1.5707963267948966*pi) q[2];
rz(-2.4594392726435226*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
cz q[9], q[10];
rz(-2.410006416418194*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.6241005045791075*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.15663442335847932*pi) q[4];
rz(-0.26702066702398364*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.4576756778947595*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[4], q[11];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.2646362487010983*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[4], q[11];
rx(1.5707963267948966*pi) q[3];
rz(-0.9054504415213644*pi) q[3];
rx(1.5707963267948966*pi) q[4];
rz(-0.405433509726161*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(2.4840098872588428*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.3687578766318635*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.5050458448537345*pi) q[5];
rz(2.1180067891878034*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(-1.4760574837720628*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.729129913085859*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.3570729334692762*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.7111478576673099*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.234551293037368*pi) q[13];
cz q[13], q[12];
rz(2.5815039033612974*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-0.2890239454553627*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.3912368115216933*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-0.6737935264688772*pi) q[12];
rz(-0.9612255677367437*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.3676392077146846*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.6554052658774783*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-2.3690717999544404*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(3.141592653589793*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.665324740486271*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.413301276748134*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.9304678397169228*pi) q[6];
rz(0.4030444784963078*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[9], q[14];
rz(1.4730068010991342*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.631729811702296*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.6562129161201913*pi) q[2];
cz q[2], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.3046923641807204*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-1.793399651144271*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.7228325125077207*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.8724766223891942*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.975915687250466*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-3.1175824342442873*pi) q[5];
cz q[5], q[4];
rz(0.5577992754800496*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(2.217146941469614*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.298264732326745*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.4364241035924823*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.789837621859456*pi) q[8];
cz q[8], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-2.4988576602910957*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(2.217146941469614*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
rz(-0.25142181235557665*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.106353299695331*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.4404601218102442*pi) q[5];
cz q[5], q[2];
rx(1.5707963267948966*pi) q[2];
rz(-2.387327264281402*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[0], q[7];
rz(0.17315129501827717*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.897732289912605*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.3312550098964409*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.3741085904801758*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.735554627433026*pi) q[8];
cz q[8], q[7];
rz(0.5177296914105547*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.1356656359639932*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.8563944222533393*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[0], q[7];
rz(1.3572636036508121*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.0636030200628968*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.760407881182692*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-3.005013399646425*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.641134363177999*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(3.0569837145072114*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(0.2427132517316307*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.261599837637768*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.4681196075215537*pi) q[2];
rz(0.5446434885353111*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(-1.1645820567151595*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.1653856061068779*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.5146654427613733*pi) q[5];
rz(0.2427132517316307*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.261599837637768*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.4681196075215537*pi) q[6];
rz(2.0634972782499377*pi) q[7];
rz(2.7695204671164304*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.0829848945212286*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.224903168164973*pi) q[8];
rz(-1.1645820567151595*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.1653856061068779*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.5146654427613733*pi) q[9];
rz(-2.415017912039522*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(3.141592653589793*pi) q[12];
rx(-1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[14];
rx(3.141592653589793*pi) q[14];
rz(2.7509805742888283*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.7507725670318017*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.483472302651915*pi) q[15];