(define (problem tireworld-091-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 - location)
    (:init (road l1 l53) (road l10 l25) (road l11 l58) (road l12 l82) (road l13 l31) (road l14 l83) (road l15 l3) (road l15 l66) (road l16 l50) (road l17 l24) (road l17 l84) (road l18 l22) (road l18 l52) (road l19 l67) (road l2 l51) (road l2 l77) (road l20 l38) (road l21 l34) (road l21 l41) (road l22 l18) (road l22 l84) (road l23 l4) (road l24 l87) (road l25 l88) (road l26 l43) (road l27 l28) (road l28 l30) (road l29 l7) (road l29 l78) (road l3 l42) (road l30 l49) (road l31 l1) (road l32 l36) (road l33 l78) (road l34 l57) (road l35 l47) (road l36 l32) (road l36 l5) (road l37 l63) (road l37 l79) (road l38 l12) (road l38 l20) (road l39 l11) (road l4 l86) (road l40 l75) (road l41 l69) (road l42 l3) (road l42 l32) (road l43 l13) (road l44 l45) (road l44 l71) (road l45 l0) (road l46 l21) (road l47 l27) (road l47 l35) (road l48 l57) (road l49 l6) (road l49 l75) (road l5 l80) (road l50 l89) (road l51 l2) (road l51 l66) (road l52 l26) (road l53 l60) (road l54 l65) (road l55 l46) (road l55 l8) (road l56 l62) (road l57 l74) (road l58 l33) (road l59 l10) (road l59 l87) (road l6 l49) (road l6 l73) (road l60 l19) (road l60 l53) (road l61 l74) (road l61 l9) (road l62 l69) (road l63 l37) (road l63 l55) (road l64 l81) (road l65 l40) (road l65 l54) (road l66 l15) (road l67 l14) (road l68 l72) (road l69 l30) (road l7 l23) (road l7 l29) (road l70 l64) (road l70 l85) (road l71 l44) (road l72 l90) (road l73 l51) (road l74 l57) (road l74 l61) (road l75 l49) (road l76 l68) (road l77 l85) (road l78 l29) (road l79 l37) (road l79 l88) (road l8 l56) (road l80 l20) (road l80 l5) (road l81 l48) (road l81 l64) (road l82 l12) (road l82 l71) (road l83 l14) (road l83 l39) (road l84 l17) (road l85 l70) (road l86 l54) (road l87 l59) (road l88 l79) (road l89 l76) (road l9 l16) (road l90 l22) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l17) (spare-in l20) (spare-in l21) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l36) (spare-in l38) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l53) (spare-in l59) (spare-in l6) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l68) (spare-in l7) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l78) (spare-in l79) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l85) (spare-in l86) (spare-in l87) (vehicle-at l35))
    (:goal (vehicle-at l0))
)