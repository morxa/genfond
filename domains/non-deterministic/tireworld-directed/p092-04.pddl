(define (problem tireworld-092-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 - location)
    (:init (road l0 l10) (road l0 l84) (road l1 l19) (road l1 l28) (road l10 l0) (road l10 l46) (road l11 l3) (road l11 l56) (road l12 l57) (road l12 l59) (road l12 l67) (road l12 l81) (road l13 l29) (road l13 l87) (road l14 l25) (road l14 l40) (road l15 l33) (road l15 l6) (road l16 l66) (road l16 l78) (road l17 l4) (road l17 l76) (road l18 l63) (road l18 l91) (road l19 l1) (road l19 l8) (road l2 l71) (road l2 l73) (road l20 l30) (road l20 l62) (road l21 l62) (road l21 l72) (road l22 l38) (road l22 l49) (road l23 l52) (road l23 l68) (road l24 l41) (road l24 l50) (road l25 l14) (road l25 l48) (road l26 l53) (road l26 l72) (road l27 l70) (road l27 l85) (road l28 l1) (road l28 l60) (road l29 l13) (road l29 l90) (road l3 l11) (road l3 l33) (road l30 l20) (road l30 l77) (road l31 l40) (road l31 l65) (road l32 l37) (road l32 l86) (road l33 l15) (road l33 l3) (road l34 l39) (road l34 l64) (road l35 l47) (road l35 l58) (road l36 l49) (road l36 l86) (road l37 l32) (road l37 l4) (road l38 l22) (road l38 l63) (road l39 l34) (road l39 l69) (road l4 l17) (road l4 l37) (road l40 l14) (road l40 l31) (road l41 l24) (road l41 l25) (road l41 l67) (road l42 l51) (road l42 l88) (road l43 l45) (road l43 l82) (road l44 l75) (road l44 l81) (road l45 l54) (road l46 l10) (road l46 l8) (road l47 l35) (road l47 l80) (road l48 l25) (road l49 l22) (road l49 l36) (road l5 l91) (road l50 l24) (road l50 l58) (road l51 l42) (road l51 l54) (road l51 l70) (road l52 l23) (road l53 l26) (road l53 l7) (road l54 l45) (road l54 l51) (road l54 l9) (road l55 l68) (road l56 l11) (road l57 l12) (road l57 l79) (road l58 l35) (road l58 l50) (road l59 l12) (road l59 l70) (road l6 l15) (road l6 l69) (road l60 l76) (road l61 l65) (road l61 l74) (road l62 l20) (road l62 l21) (road l62 l88) (road l63 l38) (road l64 l34) (road l64 l82) (road l64 l90) (road l65 l31) (road l65 l61) (road l66 l16) (road l66 l79) (road l67 l12) (road l67 l41) (road l68 l23) (road l68 l55) (road l69 l39) (road l69 l6) (road l7 l53) (road l7 l84) (road l70 l27) (road l70 l48) (road l70 l51) (road l70 l59) (road l71 l2) (road l71 l76) (road l72 l21) (road l72 l26) (road l73 l2) (road l73 l80) (road l74 l55) (road l74 l61) (road l75 l44) (road l75 l76) (road l76 l60) (road l76 l71) (road l76 l75) (road l77 l30) (road l77 l9) (road l78 l16) (road l78 l56) (road l79 l57) (road l79 l66) (road l8 l19) (road l8 l46) (road l80 l47) (road l80 l73) (road l81 l12) (road l81 l44) (road l81 l89) (road l82 l43) (road l82 l64) (road l83 l5) (road l83 l87) (road l84 l0) (road l84 l7) (road l85 l27) (road l85 l89) (road l86 l32) (road l86 l36) (road l87 l83) (road l88 l42) (road l88 l62) (road l89 l81) (road l89 l85) (road l9 l54) (road l9 l77) (road l90 l29) (road l90 l64) (road l91 l18) (road l91 l5) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l29) (spare-in l31) (spare-in l32) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l61) (spare-in l65) (spare-in l68) (spare-in l69) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l86) (spare-in l90) (vehicle-at l22))
    (:goal (vehicle-at l52))
)