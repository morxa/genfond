(define (problem tireworld-087-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l9 - location)
    (:init (road l0 l47) (road l0 l58) (road l1 l27) (road l1 l48) (road l10 l11) (road l10 l69) (road l11 l10) (road l11 l78) (road l12 l57) (road l12 l81) (road l13 l27) (road l13 l78) (road l14 l51) (road l14 l56) (road l15 l44) (road l15 l69) (road l16 l52) (road l16 l59) (road l17 l34) (road l17 l38) (road l18 l28) (road l18 l8) (road l19 l30) (road l19 l46) (road l2 l39) (road l2 l7) (road l20 l32) (road l20 l41) (road l21 l57) (road l21 l67) (road l22 l32) (road l22 l63) (road l23 l6) (road l23 l85) (road l24 l29) (road l24 l54) (road l25 l63) (road l25 l79) (road l26 l3) (road l26 l70) (road l27 l1) (road l27 l13) (road l28 l18) (road l28 l37) (road l29 l24) (road l29 l58) (road l3 l26) (road l3 l8) (road l30 l19) (road l30 l62) (road l31 l82) (road l31 l86) (road l32 l20) (road l32 l22) (road l33 l66) (road l33 l84) (road l34 l17) (road l34 l73) (road l35 l60) (road l35 l77) (road l36 l60) (road l36 l68) (road l37 l28) (road l37 l83) (road l38 l17) (road l38 l84) (road l39 l2) (road l39 l79) (road l4 l48) (road l4 l72) (road l40 l41) (road l40 l65) (road l41 l20) (road l41 l40) (road l42 l76) (road l42 l9) (road l43 l81) (road l43 l82) (road l44 l15) (road l44 l76) (road l45 l70) (road l45 l73) (road l46 l19) (road l46 l5) (road l47 l0) (road l47 l72) (road l47 l73) (road l47 l80) (road l48 l1) (road l48 l4) (road l49 l56) (road l49 l67) (road l5 l46) (road l5 l59) (road l50 l61) (road l50 l73) (road l51 l14) (road l51 l66) (road l52 l16) (road l52 l86) (road l53 l61) (road l53 l77) (road l54 l24) (road l54 l73) (road l55 l58) (road l55 l77) (road l56 l14) (road l56 l49) (road l57 l12) (road l57 l21) (road l58 l0) (road l58 l29) (road l58 l55) (road l58 l86) (road l59 l16) (road l59 l5) (road l6 l23) (road l6 l71) (road l60 l35) (road l60 l36) (road l61 l50) (road l61 l53) (road l62 l30) (road l62 l68) (road l63 l22) (road l63 l25) (road l64 l74) (road l64 l77) (road l65 l40) (road l65 l74) (road l66 l33) (road l66 l51) (road l67 l21) (road l67 l49) (road l68 l36) (road l68 l62) (road l69 l10) (road l69 l15) (road l7 l2) (road l7 l75) (road l70 l26) (road l70 l45) (road l71 l6) (road l71 l9) (road l72 l4) (road l72 l47) (road l73 l34) (road l73 l45) (road l73 l47) (road l73 l50) (road l73 l54) (road l74 l64) (road l74 l65) (road l75 l7) (road l75 l80) (road l76 l42) (road l76 l44) (road l77 l35) (road l77 l53) (road l77 l55) (road l77 l64) (road l78 l11) (road l78 l13) (road l79 l25) (road l79 l39) (road l8 l18) (road l8 l3) (road l80 l47) (road l80 l75) (road l81 l12) (road l81 l43) (road l82 l31) (road l82 l43) (road l83 l37) (road l83 l85) (road l84 l33) (road l84 l38) (road l85 l23) (road l85 l83) (road l86 l31) (road l86 l52) (road l86 l58) (road l9 l42) (road l9 l71) (spare-in l14) (spare-in l22) (spare-in l23) (spare-in l4) (spare-in l55) (spare-in l86) (vehicle-at l73))
    (:goal (vehicle-at l47))
)