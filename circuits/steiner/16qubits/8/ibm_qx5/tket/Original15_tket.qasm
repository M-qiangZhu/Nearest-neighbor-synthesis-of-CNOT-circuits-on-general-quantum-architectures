OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
u2(0,3.14159265358979) q[11];
u2(0,3.14159265358979) q[9];
u2(0,3.14159265358979) q[8];
cx q[6],q[7];
u2(0,3.14159265358979) q[4];
cx q[4],q[9];
u2(0,3.14159265358979) q[4];
u2(0,3.14159265358979) q[3];
cx q[3],q[9];
u2(0,3.14159265358979) q[9];
u2(0,3.14159265358979) q[3];
cx q[2],q[10];
cx q[2],q[5];
cx q[1],q[4];
u2(0,3.14159265358979) q[0];
cx q[8],q[0];
u2(0,3.14159265358979) q[8];
cx q[0],q[11];
u2(0,3.14159265358979) q[11];
u2(0,3.14159265358979) q[0];
