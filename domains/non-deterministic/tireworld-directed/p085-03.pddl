(define (problem tireworld-085-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l9 - location)
    (:init (road l0 l55) (road l1 l76) (road l10 l34) (road l11 l23) (road l12 l49) (road l13 l47) (road l13 l64) (road l14 l39) (road l15 l77) (road l16 l71) (road l17 l36) (road l18 l7) (road l19 l32) (road l2 l0) (road l20 l60) (road l21 l13) (road l22 l29) (road l22 l3) (road l23 l51) (road l24 l59) (road l25 l1) (road l26 l6) (road l26 l81) (road l27 l52) (road l28 l70) (road l29 l17) (road l3 l22) (road l3 l43) (road l30 l5) (road l30 l73) (road l31 l32) (road l31 l56) (road l32 l31) (road l33 l10) (road l34 l54) (road l35 l74) (road l36 l57) (road l37 l69) (road l37 l73) (road l38 l58) (road l39 l11) (road l4 l45) (road l40 l56) (road l40 l64) (road l41 l15) (road l42 l16) (road l42 l56) (road l43 l53) (road l44 l66) (road l45 l4) (road l45 l84) (road l46 l12) (road l47 l14) (road l48 l8) (road l49 l20) (road l50 l61) (road l51 l6) (road l52 l19) (road l53 l35) (road l53 l43) (road l53 l62) (road l53 l74) (road l54 l27) (road l55 l22) (road l56 l42) (road l57 l69) (road l58 l75) (road l59 l9) (road l6 l26) (road l6 l51) (road l60 l38) (road l61 l65) (road l62 l67) (road l63 l33) (road l63 l79) (road l64 l13) (road l65 l18) (road l65 l61) (road l66 l25) (road l67 l50) (road l68 l48) (road l68 l76) (road l69 l37) (road l69 l57) (road l7 l21) (road l70 l79) (road l71 l82) (road l72 l28) (road l73 l30) (road l74 l18) (road l74 l78) (road l75 l80) (road l76 l68) (road l77 l2) (road l78 l40) (road l78 l74) (road l79 l63) (road l8 l72) (road l80 l4) (road l80 l75) (road l81 l46) (road l82 l46) (road l83 l24) (road l83 l84) (road l84 l83) (road l9 l41) (road l9 l59) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l22) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l4) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l45) (spare-in l46) (spare-in l48) (spare-in l49) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l60) (spare-in l62) (spare-in l63) (spare-in l66) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l9) (vehicle-at l44))
    (:goal (vehicle-at l5))
)