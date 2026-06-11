#define OUTPUT_FILE_PREFIX "output/"

#define INPUT_FILE_PREFIX "input/"

//Define M_PI
#define M_PI 3.14159265358979323846

//Number of mesh points in simulation domain
#define I 420
#define J 420

//Number of mesh points the model was trained on
#define I_model 420
#define J_model 420

//Total number of points
#define NMAX (I*J)

//Total number of time steps in the simulation
#define TIME_STEPS 200000

//On how many steps the fields are written to a file
#define TIME_WRITE 2000

//On how many time steps to print progress
#define TIME_PRINT 5000

//How many time steps for Q relaxation
#define TIME_PRE_EVOL 20000

//How many time steps to perform before handing off to Python process
#define TIME_SYNC 10000

//How many FD steps to take for Q for each LB step
#define N_EVOL_Q 2

//Number of velocity vectors within the model -- in this case D2Q9 -- 9 velocities
#define LATTICE_VELOCITY_NUMBER 9

//Free energy elastic constant
#define L 0.1

//Constant A from free energy equation
#define A (0.1*(1-3.5/3.0))   // NOTE: Do not use spaces, otherwise the python utils will incorrectly parse this line

//Constant B from free energy equation
#define B (-0.1*3.5)

//Free energy phase parameter
#define C (0.1*3.5)

//Time step
#define DT 1

//relaxation time of the Lb scheme
//physical kinematic viscosity = (1 / 3) * (TAUF - 0.5) * dx^2 / DT
#define TAUF 1 // set to 1 because this makes LBM updates faster (see the LBM book by Kruger et al.)

//Density parameter
#define DENSITYINIT 1 //(2.0/DT)

//Molecular field coefficient
#define GAMMA 0.1

//Flow-aligning parameter
#define XI 0.8

//Ekman linear friction coefficient (damping parameter)
#define MU 0.01

//Activity
#define ALPHA 0.0035

// Geometry
#define GEOM_NAME "CROSS_JUNCTION"  // Available names: "FREE", "SINGLE_CHANNEL", "T_JUNCTION", "CROSS_JUNCTION", "CROSS_MAZE_CW", "CROSS_MAZE_CCW"

//Logical markers for bulk and boundary points
#define LMARK_BULK 2          // channel bulk
#define LMARK_OBS_BULK 4      // obstacle bulk

#define LMARK_BOT_WALL 6
#define LMARK_TOP_WALL 8
#define LMARK_LEFT_WALL 10
#define LMARK_RIGHT_WALL 12

#define LMARK_CORNER_BOT_LEFT_25 14
#define LMARK_CORNER_BOT_RIGHT_25 16
#define LMARK_CORNER_TOP_LEFT_25 18
#define LMARK_CORNER_TOP_RIGHT_25 20

#define LMARK_CORNER_BOT_LEFT_75 22
#define LMARK_CORNER_BOT_RIGHT_75 24
#define LMARK_CORNER_TOP_LEFT_75 26
#define LMARK_CORNER_TOP_RIGHT_75 28

#define LMARK_BOT_OUTLET_PBC 30
#define LMARK_TOP_OUTLET_PBC 32
#define LMARK_LEFT_OUTLET_PBC 34
#define LMARK_RIGHT_OUTLET_PBC 36

#define LMARK_BOT_OUTLET_OBC 38
#define LMARK_TOP_OUTLET_OBC 40
#define LMARK_LEFT_OUTLET_OBC 42
#define LMARK_RIGHT_OUTLET_OBC 44

// Defect detection
#define DEFECT_ORDER_THRESH 0.5

// Defect goal position fraction (overriden by environments)
#define DEFECT_GOAL_FRAC_I 0.75
#define DEFECT_GOAL_FRAC_J 0.5

// CUDA parameters
#define BLOCK_SIZE 8  // NOTE: Must be no greater than 32

// For synchronizing with Python process
#define SEND_PIPE_PATH "./c_done"
#define RECV_PIPE_PATH "./python_done"

// Visualization parameters
#define VIZ_SAMPLE_FRAC 0.03 // 0.0191 // NOTE: 0.03 for lattice side length 420, 0.0191 for length 660
#define WRITE_VIZ_DATA "TRUE"  // "TRUE" or "FALSE"

// Allow or forbid agent to create new defects
#define ALLOW_MULTIPLE_DEFECTS "TRUE"