
class InsulinModel(object):
    """Two compartment insulin model
     Source and parameters: https://iopscience.iop.org/article/10.1088/0967-3334/25/4/010/meta
     Implementation: https://github.com/ThonyPrice/Master_Thesis/tree/master
  """

    def __init__(self, ti_max):
        """Model parameters"""
        self.ti_max = ti_max  # [min] Time-to-max absorption of subcutaneously injected short-acting insulin

        # Variables - Changes each time model is updated
        self.s1_t = 0  # Insulin in compartment 1 - subcutaneous tissue
        self.s2_t = 0  # Insulin in compartment 2 - Blood plasma
        self.U_i = 0  # Insulin absorption rate (appearance of insulin in plasma)

    def get_variables(self):
        """Return vector with compartment values"""
        return [self.s1_t, self.s2_t]

    def calc_s1_roc(self, u_t, s1_t, ti_max):
        """
        Calculate Insulin RoC in C.1. [U/min] (Subcutaneous tissue)
        Keyword arguments:
        s1_t -- insulin in C.1 [U]
        u_t -- insulin input (bolus or infusion) [U]
        ti_max -- time to maximal insulin absorption [minutes]
        """
        return u_t - (s1_t / ti_max)

    def calc_s2_roc(self, s1_t, s2_t, ti_max):
        """
        Calculate Insulin RoC in C.1. [U/min] (Subcutaneous tissue)
        Keyword arguments:
        s1_t -- insulin in C.1 [Units]
        s2_t -- insulin in C.2 [Units]
        ti_max -- time to maximal insulin absorption [minutes]
        """
        return (s1_t / ti_max) - (s2_t / ti_max)

    def calc_Ui(self, s2_t, ti_max):
        """
        The insulin absorption rate [U/min] (appearance of insulin in plasma)
        Keyword arguments:
        s2_t -- insulin in C.2 [Units]
        ti_max -- time to maximal insulin absorption [minutes]
        """
        return s2_t / ti_max

    def update_compartments(self, bolus):
        """
        Given a bolus at time t, update model's compartment values
        Keyword arguments:
        bolus -- Administered bolus [Units]
        """
        self.s1_t, self.s2_t, self.U_i = self.new_values(bolus,
                                                         self.get_variables())

    def new_values(self, bolus, old_variables):
        """
        Prepare to update compartments by calc. and returning new values
        Keyword arguments:
        bolus -- Administered bolus [Units]
        """
        s1_t_old, s2_t_old = old_variables

        # Update Compartments
        s1_t = s1_t_old + self.calc_s1_roc(bolus, s1_t_old, self.ti_max)
        s2_t = s2_t_old + self.calc_s2_roc(s1_t, s2_t_old, self.ti_max)

        # Estimate appearance of insulin in plasma
        U_i = self.calc_Ui(s2_t, self.ti_max)
        return [s1_t, s2_t, U_i]


class HovorkaGlucoseModel(object):
    """
    Two compartment insulin model
         Source and parameters: https://iopscience-iop-org.focus.lib.kth.se/
                                article/10.1088/0967-3334/25/4/010/meta
    """

    def __init__(self, max_of_gluc_app=40):
        """Model params"""
        self.t_G = max_of_gluc_app  # Time of maximum glucose rate of appearance (minutes)
        self.a_G = 1  # Carbohydrate bioavailability (unitless)
        """Variables - Changes each time model is updated"""
        self.g_t = 0
        self.m_t = 0

    def get_variables(self):
        """Return vector with compartment values"""
        return [self.g_t, self.m_t]

    def set_variables(self, g_t, m_t):
        """Given vector with compartment values - Set model variables"""
        self.g_t, self.m_t = g_t, m_t
        return

    def glucose_c1(self, g_t, t_G, a_G, d_g_t=0):
        """
        Calculate RoC in Glucose C.2. (Gut)
        Keyword arguments:
        g_t -- glucose in compartment 1 already [mg]
        t_G -- time of maximum glucose rate of appearance [minutes]
        a_G -- carbohydrate bioavailability [minute]
        d_g_t -- carbohydrate intake [minute]
        """
        return -(1 / t_G) * g_t + (a_G / t_G) * d_g_t

    def glucose_c2(self, m_t, g_t, t_G):
        """
        Calculate RoC in Glucose C.2. (Plasma)
        Keyword arguments:
        m_t -- glucose in plasma (use cgm value) [mg]
        g_t -- glucose in cmopartment 1, the gut [mg]
        t_G -- time of maximum glucose rate of appearance [minutes]
        """
        return -(1 / t_G) * m_t + (1 / t_G) * g_t

    def update_compartments(self, food_glucose, max_of_gluc_app=None):
        """
        Update model's compartment values
        Keyword arguments:
        cgm -- Measured glucose value [mg/dl]
        """
        if max_of_gluc_app is None:
            max_of_gluc_app = self.t_G
        self.g_t, self.m_t = self.new_values(food_glucose, self.get_variables(), max_of_gluc_app)

    def new_values(self, food_glucose, old_variables, max_of_gluc_app):
        """
        Calculate new compartment values
        Keyword arguments:
        cgm -- Measured glucose value [mg/dl]
        """
        g_t_old, m_t_old = old_variables

        # Update Compartments
        g_t = g_t_old + self.glucose_c1(g_t_old, max_of_gluc_app, self.a_G, food_glucose)
        m_t = m_t_old + self.glucose_c2(m_t_old, g_t, max_of_gluc_app)

        # Estimate appearance of insulin in plasma
        return [g_t, m_t]

    def bergman_input(self):
        """Return the input for the Bergman Minimal Model"""
        return self.m_t
