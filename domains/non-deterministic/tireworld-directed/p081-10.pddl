(define (problem tireworld-081-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l9 - location)
    (:init (road l0 l29) (road l0 l37) (road l1 l15) (road l1 l72) (road l10 l62) (road l10 l71) (road l10 l75) (road l11 l26) (road l11 l55) (road l12 l16) (road l12 l53) (road l13 l33) (road l14 l70) (road l15 l1) (road l15 l59) (road l16 l12) (road l16 l57) (road l17 l52) (road l17 l78) (road l18 l77) (road l19 l75) (road l19 l79) (road l2 l29) (road l2 l45) (road l20 l3) (road l20 l8) (road l21 l27) (road l21 l52) (road l22 l32) (road l22 l49) (road l23 l39) (road l23 l43) (road l24 l41) (road l25 l38) (road l25 l79) (road l26 l11) (road l26 l46) (road l27 l21) (road l27 l42) (road l28 l55) (road l28 l72) (road l29 l0) (road l29 l2) (road l3 l20) (road l3 l41) (road l30 l42) (road l30 l7) (road l31 l13) (road l32 l22) (road l32 l48) (road l33 l13) (road l33 l66) (road l34 l51) (road l34 l61) (road l35 l73) (road l35 l8) (road l36 l54) (road l36 l9) (road l37 l0) (road l37 l65) (road l38 l25) (road l38 l62) (road l39 l23) (road l39 l63) (road l4 l47) (road l4 l9) (road l40 l5) (road l41 l24) (road l41 l3) (road l42 l27) (road l42 l30) (road l43 l23) (road l43 l54) (road l44 l69) (road l44 l7) (road l45 l2) (road l45 l74) (road l46 l26) (road l46 l61) (road l47 l4) (road l47 l67) (road l48 l32) (road l48 l51) (road l49 l22) (road l49 l65) (road l5 l40) (road l5 l50) (road l50 l5) (road l50 l68) (road l51 l34) (road l51 l48) (road l52 l17) (road l52 l21) (road l53 l12) (road l53 l56) (road l54 l36) (road l54 l43) (road l55 l28) (road l56 l53) (road l56 l80) (road l57 l16) (road l57 l77) (road l58 l74) (road l58 l79) (road l59 l15) (road l59 l6) (road l6 l59) (road l6 l64) (road l60 l24) (road l60 l63) (road l61 l34) (road l61 l46) (road l62 l38) (road l62 l75) (road l63 l39) (road l63 l60) (road l64 l6) (road l64 l75) (road l65 l37) (road l65 l49) (road l66 l33) (road l66 l67) (road l67 l47) (road l67 l66) (road l68 l50) (road l68 l69) (road l68 l71) (road l68 l75) (road l69 l44) (road l69 l68) (road l7 l30) (road l7 l44) (road l70 l14) (road l70 l76) (road l71 l10) (road l71 l68) (road l72 l1) (road l72 l28) (road l73 l14) (road l73 l35) (road l74 l45) (road l74 l58) (road l75 l10) (road l75 l62) (road l75 l64) (road l75 l68) (road l75 l76) (road l76 l70) (road l76 l75) (road l77 l18) (road l77 l57) (road l78 l17) (road l78 l18) (road l79 l19) (road l79 l58) (road l8 l20) (road l8 l35) (road l80 l31) (road l80 l56) (road l9 l36) (road l9 l4) (spare-in l19) (spare-in l5) (spare-in l50) (spare-in l68) (spare-in l75) (spare-in l79) (vehicle-at l25))
    (:goal (vehicle-at l40))
)