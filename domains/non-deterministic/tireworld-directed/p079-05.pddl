(define (problem tireworld-079-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l8 l9 - location)
    (:init (road l0 l18) (road l0 l56) (road l1 l9) (road l10 l16) (road l10 l63) (road l11 l38) (road l12 l77) (road l13 l43) (road l13 l7) (road l14 l62) (road l15 l35) (road l16 l6) (road l17 l45) (road l18 l29) (road l19 l58) (road l2 l26) (road l20 l53) (road l21 l4) (road l22 l72) (road l23 l73) (road l23 l74) (road l24 l8) (road l25 l36) (road l25 l46) (road l26 l2) (road l26 l57) (road l27 l4) (road l27 l40) (road l28 l31) (road l28 l71) (road l29 l65) (road l3 l63) (road l30 l5) (road l31 l28) (road l32 l31) (road l32 l76) (road l33 l24) (road l34 l67) (road l35 l66) (road l36 l25) (road l37 l2) (road l37 l76) (road l38 l70) (road l39 l36) (road l4 l27) (road l40 l14) (road l41 l45) (road l42 l78) (road l43 l13) (road l43 l55) (road l44 l75) (road l45 l17) (road l46 l19) (road l47 l50) (road l47 l8) (road l48 l44) (road l48 l6) (road l49 l73) (road l5 l30) (road l5 l51) (road l50 l41) (road l51 l64) (road l52 l60) (road l53 l12) (road l53 l20) (road l54 l15) (road l55 l43) (road l56 l0) (road l57 l39) (road l57 l45) (road l58 l21) (road l59 l42) (road l6 l48) (road l60 l69) (road l61 l30) (road l61 l69) (road l62 l52) (road l63 l10) (road l64 l51) (road l64 l56) (road l65 l20) (road l65 l29) (road l66 l1) (road l66 l35) (road l67 l22) (road l68 l3) (road l69 l60) (road l69 l61) (road l7 l49) (road l70 l55) (road l71 l37) (road l72 l22) (road l72 l54) (road l73 l23) (road l74 l23) (road l74 l33) (road l75 l32) (road l76 l37) (road l77 l34) (road l78 l11) (road l8 l24) (road l8 l47) (road l9 l59) (spare-in l10) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l25) (spare-in l26) (spare-in l28) (spare-in l3) (spare-in l31) (spare-in l32) (spare-in l37) (spare-in l44) (spare-in l45) (spare-in l48) (spare-in l50) (spare-in l57) (spare-in l6) (spare-in l62) (spare-in l63) (spare-in l70) (spare-in l71) (spare-in l75) (spare-in l78) (vehicle-at l68))
    (:goal (vehicle-at l17))
)