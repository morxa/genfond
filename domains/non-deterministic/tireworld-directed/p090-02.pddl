(define (problem tireworld-090-02)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 - location)
    (:init (road l1 l48) (road l10 l0) (road l11 l50) (road l12 l21) (road l13 l40) (road l14 l54) (road l15 l62) (road l15 l70) (road l16 l25) (road l16 l60) (road l17 l40) (road l17 l56) (road l18 l67) (road l19 l15) (road l2 l26) (road l20 l61) (road l21 l75) (road l22 l35) (road l23 l13) (road l24 l2) (road l25 l41) (road l26 l43) (road l27 l32) (road l27 l43) (road l28 l55) (road l29 l59) (road l3 l66) (road l30 l85) (road l31 l81) (road l32 l27) (road l33 l31) (road l33 l64) (road l34 l57) (road l34 l59) (road l35 l22) (road l35 l3) (road l36 l51) (road l37 l24) (road l37 l58) (road l38 l11) (road l38 l8) (road l39 l68) (road l4 l79) (road l40 l13) (road l40 l17) (road l41 l45) (road l42 l47) (road l43 l71) (road l44 l52) (road l44 l74) (road l45 l41) (road l45 l63) (road l46 l22) (road l46 l4) (road l47 l28) (road l48 l30) (road l49 l12) (road l5 l60) (road l50 l74) (road l51 l27) (road l52 l49) (road l53 l77) (road l53 l83) (road l54 l19) (road l55 l87) (road l56 l17) (road l56 l42) (road l57 l86) (road l58 l37) (road l59 l34) (road l6 l20) (road l60 l16) (road l61 l8) (road l62 l1) (road l63 l46) (road l63 l84) (road l64 l33) (road l65 l7) (road l66 l23) (road l67 l64) (road l68 l40) (road l69 l78) (road l69 l85) (road l7 l29) (road l70 l36) (road l71 l89) (road l72 l6) (road l73 l80) (road l74 l44) (road l75 l21) (road l75 l83) (road l76 l73) (road l76 l82) (road l77 l53) (road l77 l82) (road l78 l65) (road l79 l32) (road l8 l38) (road l80 l18) (road l81 l31) (road l81 l58) (road l82 l76) (road l83 l53) (road l84 l39) (road l84 l9) (road l85 l30) (road l85 l69) (road l86 l57) (road l86 l72) (road l87 l14) (road l88 l5) (road l89 l10) (road l9 l51) (road l9 l84) (spare-in l10) (spare-in l16) (spare-in l23) (spare-in l25) (spare-in l27) (spare-in l29) (spare-in l31) (spare-in l32) (spare-in l4) (spare-in l41) (spare-in l43) (spare-in l45) (spare-in l46) (spare-in l5) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l60) (spare-in l61) (spare-in l63) (spare-in l71) (spare-in l73) (spare-in l74) (spare-in l79) (spare-in l80) (spare-in l81) (spare-in l84) (spare-in l87) (spare-in l89) (spare-in l9) (vehicle-at l88))
    (:goal (vehicle-at l0))
)