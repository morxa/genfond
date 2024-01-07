(define (problem tireworld-097-01)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 l95 l96 - location)
    (:init (road l0 l47) (road l0 l90) (road l1 l87) (road l10 l22) (road l10 l84) (road l11 l28) (road l11 l51) (road l11 l70) (road l11 l82) (road l11 l96) (road l12 l59) (road l12 l60) (road l13 l54) (road l13 l9) (road l14 l88) (road l15 l19) (road l16 l2) (road l16 l86) (road l17 l26) (road l18 l30) (road l18 l96) (road l19 l15) (road l19 l47) (road l2 l16) (road l2 l68) (road l20 l35) (road l21 l17) (road l21 l8) (road l22 l10) (road l22 l30) (road l22 l81) (road l23 l15) (road l23 l5) (road l23 l74) (road l24 l59) (road l24 l69) (road l25 l49) (road l25 l7) (road l26 l36) (road l27 l29) (road l27 l4) (road l28 l11) (road l28 l68) (road l29 l27) (road l29 l78) (road l3 l6) (road l3 l76) (road l30 l14) (road l30 l18) (road l30 l22) (road l30 l41) (road l31 l50) (road l31 l86) (road l32 l35) (road l32 l76) (road l33 l46) (road l33 l63) (road l34 l43) (road l35 l20) (road l35 l32) (road l36 l26) (road l36 l55) (road l37 l70) (road l37 l71) (road l38 l55) (road l38 l80) (road l39 l67) (road l39 l7) (road l4 l27) (road l4 l64) (road l40 l45) (road l40 l66) (road l40 l81) (road l41 l30) (road l41 l57) (road l42 l61) (road l42 l65) (road l43 l34) (road l43 l49) (road l43 l90) (road l44 l73) (road l44 l75) (road l45 l40) (road l45 l56) (road l46 l33) (road l46 l69) (road l47 l0) (road l47 l19) (road l48 l72) (road l48 l80) (road l49 l25) (road l49 l43) (road l5 l23) (road l5 l66) (road l50 l79) (road l51 l11) (road l51 l84) (road l52 l95) (road l53 l9) (road l53 l93) (road l54 l13) (road l54 l58) (road l55 l38) (road l56 l45) (road l56 l83) (road l57 l41) (road l57 l94) (road l58 l54) (road l58 l56) (road l59 l12) (road l59 l24) (road l6 l3) (road l6 l82) (road l60 l12) (road l60 l91) (road l61 l42) (road l61 l83) (road l62 l67) (road l62 l95) (road l63 l77) (road l64 l4) (road l64 l88) (road l65 l42) (road l65 l73) (road l66 l5) (road l67 l62) (road l68 l2) (road l69 l24) (road l69 l46) (road l69 l71) (road l7 l25) (road l7 l39) (road l70 l11) (road l70 l37) (road l71 l37) (road l71 l69) (road l72 l30) (road l72 l48) (road l73 l44) (road l73 l65) (road l74 l23) (road l74 l9) (road l75 l44) (road l75 l93) (road l76 l3) (road l77 l63) (road l77 l85) (road l78 l29) (road l78 l89) (road l79 l50) (road l79 l91) (road l8 l21) (road l8 l86) (road l80 l38) (road l80 l48) (road l81 l22) (road l81 l40) (road l82 l11) (road l82 l6) (road l83 l56) (road l83 l61) (road l84 l10) (road l84 l51) (road l85 l77) (road l85 l92) (road l86 l16) (road l86 l31) (road l86 l8) (road l87 l1) (road l87 l41) (road l88 l14) (road l88 l64) (road l89 l1) (road l89 l78) (road l89 l93) (road l9 l13) (road l9 l53) (road l9 l74) (road l90 l43) (road l91 l60) (road l91 l79) (road l92 l85) (road l92 l94) (road l93 l53) (road l93 l75) (road l93 l89) (road l94 l57) (road l94 l92) (road l95 l52) (road l95 l62) (road l96 l11) (road l96 l18) (road l96 l34) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l44) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l79) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l88) (spare-in l89) (spare-in l9) (spare-in l90) (spare-in l91) (spare-in l92) (spare-in l93) (spare-in l94) (spare-in l95) (vehicle-at l20))
    (:goal (vehicle-at l52))
)