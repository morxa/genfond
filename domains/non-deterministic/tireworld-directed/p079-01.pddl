(define (problem tireworld-079-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l8 l9 - location)
    (:init (road l0 l23) (road l1 l17) (road l10 l64) (road l10 l66) (road l11 l1) (road l11 l42) (road l13 l16) (road l13 l72) (road l14 l33) (road l15 l24) (road l16 l13) (road l17 l31) (road l18 l27) (road l19 l28) (road l19 l35) (road l2 l47) (road l2 l9) (road l20 l58) (road l21 l53) (road l21 l73) (road l22 l26) (road l22 l5) (road l23 l6) (road l24 l48) (road l25 l64) (road l26 l22) (road l26 l67) (road l27 l16) (road l27 l18) (road l27 l42) (road l28 l56) (road l29 l0) (road l29 l63) (road l3 l54) (road l30 l75) (road l31 l68) (road l32 l44) (road l32 l45) (road l33 l20) (road l34 l53) (road l34 l62) (road l35 l7) (road l36 l78) (road l37 l59) (road l37 l72) (road l38 l27) (road l38 l55) (road l39 l36) (road l4 l43) (road l40 l63) (road l41 l52) (road l42 l11) (road l42 l27) (road l43 l67) (road l44 l50) (road l44 l74) (road l45 l32) (road l45 l49) (road l46 l39) (road l47 l2) (road l48 l52) (road l49 l30) (road l49 l45) (road l5 l50) (road l5 l70) (road l50 l44) (road l50 l76) (road l51 l25) (road l52 l18) (road l52 l48) (road l53 l34) (road l54 l19) (road l55 l38) (road l56 l28) (road l56 l69) (road l57 l55) (road l58 l56) (road l59 l37) (road l59 l40) (road l6 l4) (road l60 l71) (road l61 l14) (road l62 l34) (road l62 l46) (road l63 l29) (road l64 l10) (road l64 l12) (road l64 l41) (road l65 l57) (road l66 l10) (road l66 l52) (road l67 l26) (road l68 l32) (road l69 l27) (road l7 l66) (road l70 l73) (road l71 l60) (road l71 l61) (road l72 l37) (road l73 l21) (road l73 l70) (road l74 l77) (road l75 l3) (road l76 l15) (road l76 l50) (road l77 l47) (road l78 l8) (road l8 l60) (road l9 l2) (road l9 l51) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l34) (spare-in l36) (spare-in l38) (spare-in l39) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l60) (spare-in l64) (spare-in l68) (spare-in l7) (spare-in l72) (spare-in l74) (spare-in l75) (spare-in l77) (spare-in l9) (vehicle-at l65))
    (:goal (vehicle-at l12))
)