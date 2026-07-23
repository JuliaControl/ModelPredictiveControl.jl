#include "types.h"
#include "constants.h"
// Settings
DAQPSettings settings = {(c_float)0.00000100000000000000, (c_float)0.00000000000100000000, (c_float)0.00000000001000000000, (c_float)0.00000100000000000000, (c_float)0.00000000000001000000, 10, 1000, (c_float)1000000000000000019884624838656.00000000000000000000, (c_float)0.00000000000000000000, (c_float)0.00000100000000000000, (c_float)0.00010000000000000000,(c_float)0.00000000000000000000,(c_float)0.00000000000000000000,(c_float)0.00000000003700000000,(c_float)0.00000000100000000000};

// Workspace
c_float M[40] = {
(c_float)0.67968822373856518926,
(c_float)0.52719873051750865134,
(c_float)-0.41872125210590377398,
(c_float)-0.29113215227230732074,
(c_float)0.75289580297325875424,
(c_float)0.58398203422770433857,
(c_float)-0.24918805225706575079,
(c_float)-0.17325763526279078230,
(c_float)0.79011090488375301799,
(c_float)0.61284784916764190044,
(c_float)-0.00979323884155458395,
(c_float)-0.00680912823822322176,
(c_float)0.75020905756604516768,
(c_float)0.58189806584567815850,
(c_float)0.25778855766515101910,
(c_float)0.17923746943049204128,
(c_float)0.64319623422893668074,
(c_float)0.49889379617906376430,
(c_float)0.47691595187263807754,
(c_float)0.33159426903547800647,
(c_float)0.52068316369708089741,
(c_float)0.40386679262009111957,
(c_float)0.61757691072387899123,
(c_float)0.42939424332646730642,
(c_float)0.41511375109173059839,
(c_float)0.32198210142904992725,
(c_float)0.69861630182833933667,
(c_float)0.48574001568079633318,
(c_float)0.33272501864973985652,
(c_float)0.25807745568801804259,
(c_float)0.74470648470815437037,
(c_float)0.51778599871351582706,
(c_float)0.26993020959037150597,
(c_float)0.20937079510017478357,
(c_float)0.77165202413054656549,
(c_float)0.53652092761126202181,
(c_float)0.22184451227729398703,
(c_float)0.17207322587047085838,
(c_float)0.78802186360017267841,
(c_float)0.54790269190714002701,
};
c_float dupper[10];
c_float dlower[10];
c_float Rinv[10] = {
(c_float)1.49692065038253452336,
(c_float)-0.38440691960414097306,
(c_float)-0.62837362292457599189,
(c_float)-0.07999098122961971480,
(c_float)1.54549025017345686983,
(c_float)-0.29380285025858715597,
(c_float)-0.56118789388703360643,
(c_float)0.38186052108230555957,
(c_float)-0.14185408053419068519,
(c_float)0.40735738329561493876,
};
int sense[10] = {
(int)8,
(int)8,
(int)8,
(int)8,
(int)8,
(int)8,
(int)8,
(int)8,
(int)8,
(int)8,
};
c_float x[5];
c_float xold[5];

c_float lam[15];
c_float lam_star[15];
c_float u[5];

c_float L[120];
c_float D[15];
c_float xldl[15];
c_float zldl[15];

int WS[15];

DAQPWorkspace daqp_work= {
NULL,
4, 10, 0,
M, dupper, dlower, Rinv, NULL, sense,
NULL,
NULL,
x, xold,
lam, lam_star, u, -1,
L, D, xldl,zldl,0,
WS, 0,
0,-1,
0.000000,
&settings, 
NULL,
0, NULL};

c_float mpc_parameter[11] = {
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
};
c_float Dth[110] = {
(c_float)0.00001170403560364362,
(c_float)0.00000000000000000006,
(c_float)0.74705779833475449703,
(c_float)0.00000000000000001683,
(c_float)-1.09741131731677987737,
(c_float)-0.00000000000000010759,
(c_float)-1.17508907023199138386,
(c_float)0.00000000000000010759,
(c_float)-0.21412701492452229646,
(c_float)-0.11011818346276878999,
(c_float)-0.11011818346276873448,
(c_float)0.00003565257464074624,
(c_float)0.00000000000000000001,
(c_float)0.71343152151789168691,
(c_float)0.00000000000000000274,
(c_float)-1.04916234588373380454,
(c_float)-0.00000000000000002645,
(c_float)-1.76393032442441399787,
(c_float)0.00000000000000002645,
(c_float)-0.20676015421935267646,
(c_float)-0.10729807876118648002,
(c_float)-0.10729807876118642451,
(c_float)0.00005344088036330545,
(c_float)-0.00000000000000000000,
(c_float)0.61490306463047716967,
(c_float)-0.00000000000000000531,
(c_float)-0.90581801112271265186,
(c_float)0.00000000000000010610,
(c_float)-2.39325678001189734445,
(c_float)-0.00000000000000010610,
(c_float)-0.18123588625858277901,
(c_float)-0.09538488575484913023,
(c_float)-0.09538488575484908860,
(c_float)0.00005770109076596212,
(c_float)-0.00000000000000000004,
(c_float)0.44189093655980948849,
(c_float)-0.00000000000000000729,
(c_float)-0.65294554517713898623,
(c_float)0.00000000000000018681,
(c_float)-2.84764528728663224300,
(c_float)-0.00000000000000018681,
(c_float)-0.13410114797878194826,
(c_float)-0.07229928763778772915,
(c_float)-0.07229928763778768752,
(c_float)0.00004752602601552108,
(c_float)-0.00000000000000000008,
(c_float)0.24285146624465783760,
(c_float)-0.00000000000000001502,
(c_float)-0.36135576405022445945,
(c_float)0.00000000000000018538,
(c_float)-2.99260118169340305627,
(c_float)-0.00000000000000018538,
(c_float)-0.07853474191894452283,
(c_float)-0.04448295424077684107,
(c_float)-0.04448295424077681331,
(c_float)0.00002974376097342301,
(c_float)-0.00000000000000000012,
(c_float)0.07355948615563938364,
(c_float)-0.00000000000000003158,
(c_float)-0.11299584651664298929,
(c_float)0.00000000000000050629,
(c_float)-2.92119202268920163945,
(c_float)-0.00000000000000050629,
(c_float)-0.03057436570440456164,
(c_float)-0.02017554826790108474,
(c_float)-0.02017554826790110209,
(c_float)0.00001037871615635384,
(c_float)-0.00000000000000000007,
(c_float)-0.05096769872824821745,
(c_float)-0.00000000000000000125,
(c_float)0.06986521121833964842,
(c_float)0.00000000000000004994,
(c_float)-2.77314498657948727001,
(c_float)-0.00000000000000004994,
(c_float)0.00505704394229119277,
(c_float)-0.00197733213981173299,
(c_float)-0.00197733213981174296,
(c_float)-0.00000807853622577395,
(c_float)-0.00000000000000000009,
(c_float)-0.13903169003536494652,
(c_float)0.00000000000000000454,
(c_float)0.19927047819551862995,
(c_float)0.00000000000000021208,
(c_float)-2.62065786416566748329,
(c_float)-0.00000000000000021208,
(c_float)0.03044073454986001917,
(c_float)0.01105159303601366792,
(c_float)0.01105159303601366966,
(c_float)-0.00002503241487870051,
(c_float)-0.00000000000000000000,
(c_float)-0.20180076150140752178,
(c_float)0.00000000000000001400,
(c_float)0.29155399834135303783,
(c_float)-0.00000000000000029459,
(c_float)-2.48681039816054427760,
(c_float)0.00000000000000029459,
(c_float)0.04863834109124191146,
(c_float)0.02042220909914597635,
(c_float)0.02042220909914596594,
(c_float)-0.00004051321323676852,
(c_float)-0.00000000000000000009,
(c_float)-0.24759883289234116410,
(c_float)-0.00000000000000000557,
(c_float)0.35891423209081929624,
(c_float)0.00000000000000009223,
(c_float)-2.37513179080281267019,
(c_float)-0.00000000000000009223,
(c_float)0.06197988385211233853,
(c_float)0.02730619344195108975,
(c_float)0.02730619344195107240,
};
c_float du[10] = {
(c_float)2272500387548771198163017007104.00000000000000000000,
(c_float)2813092670308147670282041032704.00000000000000000000,
(c_float)3299074791134610415875749052416.00000000000000000000,
(c_float)3500590832463771595082924294144.00000000000000000000,
(c_float)3353956945743627794796213436416.00000000000000000000,
(c_float)3034187869205844792218904166400.00000000000000000000,
(c_float)2703279775361147475930973732864.00000000000000000000,
(c_float)2421387385970148857912246665216.00000000000000000000,
(c_float)2195256399819191174282652680192.00000000000000000000,
(c_float)2016217558711993387514389856256.00000000000000000000,
};
c_float dl[10] = {
(c_float)58.35171959840567978972,
(c_float)85.37127177482263107322,
(c_float)113.90665299142204958116,
(c_float)133.95389549956371411099,
(c_float)139.56424430970562866605,
(c_float)135.34135890244087363499,
(c_float)127.82208263427642691568,
(c_float)120.29646525196503148436,
(c_float)113.76983912295973766504,
(c_float)108.35987389057237351153,
};
c_float Uth_offset[22] = {
(c_float)0.00007233190847382115,
(c_float)-0.00027712431861415460,
(c_float)0.78668677603934666909,
(c_float)-0.25677001857163778142,
(c_float)-1.29398673820900778075,
(c_float)1.29402865173813097499,
(c_float)1.29398673820900778075,
(c_float)-1.29402865173813097499,
(c_float)-0.07663972354151854516,
(c_float)0.27897920804466486144,
(c_float)-0.03645892146186489186,
(c_float)0.00007233190847382104,
(c_float)0.00027712431861415428,
(c_float)0.78668677603934622500,
(c_float)0.25677001857163789245,
(c_float)-1.29398673820900711462,
(c_float)-1.29402865173812853250,
(c_float)1.29398673820900711462,
(c_float)1.29402865173812853250,
(c_float)-0.90055353952386762995,
(c_float)-0.03645892146186489186,
(c_float)0.27897920804466491695,
};
c_float u_offset[2] = {
(c_float)-9.19608861913209452155,
(c_float)-70.35953140377290537799,
};
c_float uscaling[2] = {
(c_float)1.00000000000000000000,
(c_float)1.00000000000000000000,
};
#include "mpc_funcs.h"
#if N_LINEAR_COST > 0
void mpc_update_parameter(c_float* parameter, c_float* control, c_float* state, c_float* reference, c_float* disturbance, c_float* linear_cost){
#else
void mpc_update_parameter(c_float* parameter, c_float* control, c_float* state, c_float* reference, c_float* disturbance){
#endif
    int i,j;
    // update parameter
    for(i=0,j=0;j<N_STATE;i++, j++) parameter[i] = state[j];
#ifdef N_PREVIEW_HORIZON
    int disp = 0;
    for(;i<N_REFERENCE;i++) parameter[i] = 0.0; // reset parameter
    for(i=0;i<N_REFERENCE*N_PREVIEW_HORIZON;i++)
        for(j=0;j<N_REFERENCE;j++)
            parameter[N_STATE+j] += reference[i]*traj2setpoint[disp++];
    i = N_STATE + N_REFERENCE; // Setup i for remaining parameters
#else
    for(j=0;j<N_REFERENCE;i++, j++) parameter[i] = reference[j];
#endif
    for(j=0;j<N_DISTURBANCE;i++, j++) parameter[i] = disturbance[j];
    for(j=0;j<N_CONTROL_PREV;i++, j++) parameter[i] = control[j];
#if N_LINEAR_COST > 0
#ifdef N_MOVE_BLOCKS
    // Average linear cost over move blocks
    // linear_cost is (N_CONTROL x N_PREDICTION_HORIZON), column-major
    int block_offset = 0;
    for(int b = 0; b < N_MOVE_BLOCKS; b++) {
        int block_size = move_blocks[b];
        for(int u = 0; u < N_CONTROL; u++) {
            c_float sum = 0.0;
            for(int k = 0; k < block_size; k++) {
                sum += linear_cost[u + (block_offset + k) * N_CONTROL];
            }
            parameter[i++] = sum / block_size;
        }
        block_offset += block_size;
    }
#else
    for(j=0;j<N_LINEAR_COST;i++, j++) parameter[i] = linear_cost[j];
#endif
#endif
}
void mpc_update_qp(c_float* th, c_float* dupper, c_float* dlower){
    int i,j,disp;
    c_float b_shift_th;
    for(i =0,disp=0; i < N_CONSTR; i++){
        b_shift_th = 0;
        for(j = 0; j < N_THETA; j++) b_shift_th += Dth[disp++]*th[j];
        dupper[i] = du[i] + b_shift_th;
        dlower[i] = dl[i] + b_shift_th;
    }
}

// Assumes that x is stacked such that the first
// N_CONTROL elements are the controls at the first time step
void mpc_get_solution(c_float* th, c_float* control, c_float* xstar){
    int i,j,disp;
    c_float ctr_shift_th;
    for(i = 0, disp=0; i < N_CONTROL; i++){
        ctr_shift_th = u_offset[i];
        for(j = 0; j < N_THETA; j++) ctr_shift_th += Uth_offset[disp++]*th[j];
        control[i] = uscaling[i]*xstar[i]+ctr_shift_th;
    }
}

#include"daqp.h"
#ifdef DAQP_BNB
#include "bnb.h"
#endif

#if N_LINEAR_COST > 0
int mpc_compute_control(c_float* control, c_float* state, c_float* reference, c_float* disturbance, c_float* linear_cost){
    mpc_update_parameter(mpc_parameter, control, state, reference, disturbance, linear_cost);
#else
int mpc_compute_control(c_float* control, c_float* state, c_float* reference, c_float* disturbance){
    mpc_update_parameter(mpc_parameter, control, state, reference, disturbance);
#endif
    // update problem
    mpc_update_qp(mpc_parameter,daqp_work.dupper,daqp_work.dlower);
    daqp_work.reuse_ind=0; // clear workspace cache

#ifdef DAQP_BNB
    daqp_node_cleanup_workspace(0, &daqp_work);
    int exitflag = daqp_bnb(&daqp_work);
#else
#ifndef DAQP_WARMSTART
    daqp_deactivate_constraints(&daqp_work);
    reset_daqp_workspace(&daqp_work);
#endif
    int exitflag = daqp_ldp(&daqp_work);
#endif

    ldp2qp_solution(&daqp_work);
    mpc_get_solution(mpc_parameter,control,daqp_work.x);
    return exitflag;
}
c_float MPC_PLANT_DYNAMICS[60] = {
(c_float)-0.42151435483677968108,
(c_float)0.89476858379336154936,
(c_float)0.00000000000000000000,
(c_float)0.00004738338834550184,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.18623340238006225178,
(c_float)-0.18623340238006230729,
(c_float)0.39354252250196353202,
(c_float)-15.05153389413755427029,
(c_float)0.00000000000000000000,
(c_float)0.77806747839190204541,
(c_float)0.00000000000000000000,
(c_float)0.00046091085471307986,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.33293875109574344595,
(c_float)-0.33293875109574344595,
(c_float)0.75257669470687771351,
(c_float)16.39283625966390900430,
(c_float)0.00004738338834552434,
(c_float)0.00000000000000000000,
(c_float)0.89480757512627184802,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.27800568138014608577,
(c_float)-0.27800568138014608577,
(c_float)-0.26363045022290315655,
(c_float)9.46048149671451454878,
(c_float)0.00000000000000000000,
(c_float)0.00046091085471298179,
(c_float)0.00000000000000000000,
(c_float)0.77851108245728029011,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.52970230939405582582,
(c_float)-0.52970230939405582582,
(c_float)-0.47302407483572572744,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
};
c_float MPC_MEASUREMENT_FUNCTION[16] = {
(c_float)48.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.71870940572427688409,
(c_float)0.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.09999999999999999167,
(c_float)28.35555555555555429237,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)-0.30901775888877647258,
(c_float)0.00000000000000000000,
(c_float)1.00000000000000000000,
(c_float)0.08222222222222222432,
};
c_float K_TRANSPOSE_OBSERVER[12] = {
(c_float)-0.00000005355876773996,
(c_float)0.00000000000000000000,
(c_float)-0.01355901959025781860,
(c_float)0.00000000000000000000,
(c_float)0.61366736734452342894,
(c_float)0.00000000000000000000,
(c_float)0.00000000000000000000,
(c_float)0.00000004481237662412,
(c_float)0.00000000000000000000,
(c_float)-0.00588796749927334405,
(c_float)0.00000000000000000000,
(c_float)0.61721999332224852797,
};
void mpc_predict_state(c_float* state, c_float* control, c_float* disturbance){
    int i,j,disp=0;
    c_float state_old[N_STATE];
    for(i=0;i<N_STATE;i++) state_old[i] = state[i];
    for(i=0,disp=0;i<N_STATE;i++){
        state[i] = MPC_PLANT_DYNAMICS[disp++];
        for(j=0;j<N_STATE;j++) state[i] += MPC_PLANT_DYNAMICS[disp++]*state_old[j];
        for(j=0;j<N_CONTROL;j++) state[i] += MPC_PLANT_DYNAMICS[disp++]*control[j];
        for(j=0;j<N_DISTURBANCE;j++) state[i] += MPC_PLANT_DYNAMICS[disp++]*disturbance[j];
    }
}

void mpc_correct_state(c_float* state, c_float* measurement, c_float* disturbance){
    int i,j,disp_C=0,disp_K=0;
    c_float innovation, state_old[N_STATE];
    for(i=0;i<N_STATE;i++) state_old[i] = state[i];
    for(j=0;j<N_MEASUREMENT;j++){
        innovation = measurement[j]-MPC_MEASUREMENT_FUNCTION[disp_C++];
        for(i=0;i<N_STATE;i++) innovation -= MPC_MEASUREMENT_FUNCTION[disp_C++]*state_old[i];
        for(i=0;i<N_DISTURBANCE;i++) innovation -= MPC_MEASUREMENT_FUNCTION[disp_C++]*disturbance[i];
        for(i=0;i<N_STATE;i++) state[i] += K_TRANSPOSE_OBSERVER[disp_K++]*innovation;
    }
}
