(define (problem tireworld-085-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l9 - location)
    (:init (road l0 l45) (road l1 l48) (road l1 l71) (road l10 l32) (road l11 l58) (road l12 l26) (road l13 l4) (road l13 l62) (road l14 l11) (road l15 l44) (road l16 l53) (road l17 l19) (road l18 l73) (road l19 l25) (road l2 l3) (road l2 l33) (road l20 l67) (road l21 l17) (road l21 l6) (road l22 l79) (road l23 l84) (road l24 l81) (road l25 l22) (road l25 l80) (road l26 l16) (road l27 l65) (road l28 l55) (road l28 l67) (road l29 l54) (road l3 l9) (road l30 l59) (road l31 l82) (road l32 l42) (road l32 l56) (road l33 l2) (road l34 l63) (road l35 l78) (road l36 l83) (road l37 l76) (road l38 l27) (road l39 l61) (road l4 l70) (road l40 l31) (road l41 l29) (road l42 l12) (road l42 l32) (road l43 l71) (road l43 l74) (road l44 l18) (road l44 l55) (road l45 l52) (road l46 l49) (road l46 l70) (road l47 l36) (road l48 l69) (road l49 l41) (road l49 l46) (road l5 l60) (road l50 l59) (road l51 l7) (road l52 l45) (road l52 l9) (road l53 l16) (road l53 l42) (road l54 l66) (road l55 l10) (road l55 l44) (road l56 l30) (road l57 l6) (road l57 l8) (road l58 l11) (road l58 l23) (road l59 l72) (road l6 l21) (road l6 l57) (road l60 l77) (road l61 l37) (road l61 l39) (road l62 l13) (road l63 l0) (road l64 l20) (road l64 l80) (road l65 l14) (road l66 l24) (road l66 l54) (road l67 l20) (road l67 l28) (road l68 l40) (road l68 l78) (road l69 l39) (road l69 l48) (road l7 l34) (road l7 l51) (road l70 l4) (road l70 l46) (road l71 l1) (road l72 l57) (road l72 l59) (road l73 l26) (road l74 l43) (road l74 l79) (road l75 l62) (road l76 l33) (road l76 l37) (road l77 l47) (road l77 l60) (road l78 l68) (road l79 l22) (road l79 l74) (road l8 l75) (road l80 l64) (road l81 l24) (road l81 l35) (road l82 l31) (road l82 l5) (road l83 l36) (road l83 l51) (road l84 l23) (road l84 l50) (road l9 l38) (spare-in l0) (spare-in l10) (spare-in l18) (spare-in l21) (spare-in l22) (spare-in l26) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l34) (spare-in l35) (spare-in l41) (spare-in l43) (spare-in l44) (spare-in l47) (spare-in l48) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l63) (spare-in l64) (spare-in l66) (spare-in l69) (spare-in l71) (spare-in l72) (spare-in l74) (spare-in l77) (spare-in l81) (vehicle-at l15))
    (:goal (vehicle-at l32))
)