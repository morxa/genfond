(define (problem tireworld-076-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l8 l9 - location)
    (:init (road l0 l2) (road l0 l30) (road l1 l62) (road l1 l68) (road l10 l14) (road l11 l22) (road l11 l42) (road l12 l2) (road l12 l62) (road l13 l19) (road l13 l30) (road l13 l41) (road l14 l10) (road l14 l36) (road l15 l27) (road l15 l49) (road l16 l41) (road l16 l54) (road l17 l26) (road l17 l73) (road l18 l67) (road l19 l13) (road l19 l31) (road l2 l0) (road l2 l12) (road l20 l31) (road l20 l72) (road l21 l70) (road l21 l73) (road l22 l11) (road l22 l60) (road l23 l29) (road l23 l52) (road l24 l8) (road l25 l60) (road l26 l17) (road l26 l53) (road l27 l15) (road l27 l37) (road l28 l33) (road l28 l36) (road l28 l55) (road l29 l23) (road l29 l5) (road l3 l32) (road l30 l0) (road l31 l19) (road l31 l20) (road l32 l25) (road l32 l3) (road l33 l12) (road l33 l28) (road l34 l3) (road l34 l7) (road l35 l18) (road l35 l53) (road l36 l28) (road l37 l46) (road l38 l50) (road l38 l56) (road l39 l64) (road l39 l66) (road l39 l9) (road l4 l55) (road l4 l9) (road l40 l10) (road l41 l16) (road l42 l11) (road l42 l49) (road l43 l51) (road l43 l68) (road l44 l47) (road l44 l59) (road l45 l71) (road l46 l37) (road l46 l52) (road l47 l24) (road l48 l2) (road l48 l74) (road l49 l15) (road l49 l42) (road l5 l29) (road l5 l58) (road l50 l38) (road l50 l75) (road l51 l43) (road l51 l45) (road l52 l23) (road l52 l46) (road l53 l35) (road l54 l16) (road l54 l65) (road l55 l28) (road l55 l4) (road l56 l38) (road l57 l72) (road l58 l5) (road l58 l6) (road l59 l44) (road l59 l64) (road l59 l71) (road l6 l58) (road l6 l74) (road l60 l22) (road l60 l25) (road l61 l63) (road l61 l69) (road l62 l1) (road l62 l12) (road l63 l61) (road l64 l59) (road l65 l54) (road l65 l63) (road l66 l39) (road l66 l70) (road l67 l7) (road l68 l1) (road l68 l43) (road l69 l56) (road l7 l34) (road l70 l21) (road l70 l66) (road l71 l45) (road l71 l59) (road l72 l20) (road l72 l57) (road l73 l17) (road l73 l21) (road l74 l48) (road l74 l6) (road l75 l40) (road l75 l50) (road l9 l39) (road l9 l4) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l16) (spare-in l19) (spare-in l20) (spare-in l24) (spare-in l28) (spare-in l29) (spare-in l31) (spare-in l33) (spare-in l36) (spare-in l38) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l47) (spare-in l48) (spare-in l50) (spare-in l51) (spare-in l53) (spare-in l54) (spare-in l56) (spare-in l59) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l65) (spare-in l68) (spare-in l69) (spare-in l71) (spare-in l72) (spare-in l75) (vehicle-at l57))
    (:goal (vehicle-at l8))
)