OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
u2(0,3.14159265358979) q[9];
cx q[8],q[6];
u2(0,3.14159265358979) q[6];
cx q[6],q[9];
u2(0,3.14159265358979) q[9];
u2(0,3.14159265358979) q[6];
u2(0,3.14159265358979) q[5];
cx q[7],q[4];
cx q[2],q[1];
cx q[10],q[1];
cx q[10],q[3];
cx q[1],q[8];
u2(0,3.14159265358979) q[0];
cx q[5],q[0];
u2(0,3.14159265358979) q[5];
u2(0,3.14159265358979) q[0];
