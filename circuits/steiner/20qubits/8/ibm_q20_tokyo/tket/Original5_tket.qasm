OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
u2(0,3.14159265358979) q[12];
u2(0,3.14159265358979) q[11];
cx q[13],q[10];
u2(0,3.14159265358979) q[8];
cx q[12],q[8];
u2(0,3.14159265358979) q[8];
u2(0,3.14159265358979) q[12];
u2(0,3.14159265358979) q[7];
cx q[11],q[7];
u2(0,3.14159265358979) q[7];
u2(0,3.14159265358979) q[11];
u2(0,3.14159265358979) q[6];
cx q[5],q[9];
u2(0,3.14159265358979) q[5];
u2(0,3.14159265358979) q[4];
cx q[6],q[4];
u2(0,3.14159265358979) q[4];
u2(0,3.14159265358979) q[6];
u2(0,3.14159265358979) q[3];
u2(0,3.14159265358979) q[2];
cx q[2],q[5];
u2(0,3.14159265358979) q[5];
u2(0,3.14159265358979) q[2];
u2(0,3.14159265358979) q[1];
u2(0,3.14159265358979) q[0];
cx q[1],q[0];
cx q[3],q[1];
u2(0,3.14159265358979) q[1];
u2(0,3.14159265358979) q[3];
u2(0,3.14159265358979) q[0];
