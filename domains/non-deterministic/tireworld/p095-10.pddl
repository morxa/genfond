(define (problem tireworld-095-10)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 l92 l93 l94 - location)
    (:init (road l0 l54) (road l0 l84) (road l1 l13) (road l1 l82) (road l10 l59) (road l10 l64) (road l11 l34) (road l11 l67) (road l12 l18) (road l12 l22) (road l12 l46) (road l13 l1) (road l13 l56) (road l14 l70) (road l14 l90) (road l15 l38) (road l15 l76) (road l16 l20) (road l16 l37) (road l17 l25) (road l17 l54) (road l18 l12) (road l18 l41) (road l19 l83) (road l19 l85) (road l2 l22) (road l20 l16) (road l20 l63) (road l21 l5) (road l21 l68) (road l21 l77) (road l22 l12) (road l22 l2) (road l22 l81) (road l23 l76) (road l23 l84) (road l24 l33) (road l24 l4) (road l24 l61) (road l24 l71) (road l24 l87) (road l25 l17) (road l25 l55) (road l26 l32) (road l26 l88) (road l27 l67) (road l27 l88) (road l28 l53) (road l28 l57) (road l28 l79) (road l29 l43) (road l29 l90) (road l3 l35) (road l3 l71) (road l30 l43) (road l30 l91) (road l31 l34) (road l31 l58) (road l32 l26) (road l32 l80) (road l33 l24) (road l33 l61) (road l34 l11) (road l34 l31) (road l34 l4) (road l35 l3) (road l35 l52) (road l36 l66) (road l36 l72) (road l37 l16) (road l37 l6) (road l38 l15) (road l38 l60) (road l39 l48) (road l39 l69) (road l4 l24) (road l4 l34) (road l40 l89) (road l40 l9) (road l41 l18) (road l41 l61) (road l42 l62) (road l42 l72) (road l43 l29) (road l43 l30) (road l44 l63) (road l45 l49) (road l45 l75) (road l46 l12) (road l46 l69) (road l47 l78) (road l47 l9) (road l48 l39) (road l48 l55) (road l49 l45) (road l49 l73) (road l5 l21) (road l5 l81) (road l50 l52) (road l50 l56) (road l51 l68) (road l51 l85) (road l52 l35) (road l52 l50) (road l53 l28) (road l53 l70) (road l54 l0) (road l54 l17) (road l55 l25) (road l55 l48) (road l56 l13) (road l56 l50) (road l57 l28) (road l57 l7) (road l57 l87) (road l58 l31) (road l58 l91) (road l59 l10) (road l59 l8) (road l6 l37) (road l6 l86) (road l60 l38) (road l60 l79) (road l61 l24) (road l61 l33) (road l61 l41) (road l61 l92) (road l62 l42) (road l62 l74) (road l63 l20) (road l63 l44) (road l64 l10) (road l64 l71) (road l65 l74) (road l65 l78) (road l66 l36) (road l66 l80) (road l67 l11) (road l67 l27) (road l68 l21) (road l68 l51) (road l69 l39) (road l69 l46) (road l7 l57) (road l7 l92) (road l70 l14) (road l70 l53) (road l71 l24) (road l71 l3) (road l71 l64) (road l72 l36) (road l72 l42) (road l73 l49) (road l73 l83) (road l74 l62) (road l74 l65) (road l75 l45) (road l75 l94) (road l76 l15) (road l76 l23) (road l77 l21) (road l77 l92) (road l78 l47) (road l78 l65) (road l79 l28) (road l79 l60) (road l8 l59) (road l8 l93) (road l80 l32) (road l80 l66) (road l80 l93) (road l81 l22) (road l81 l5) (road l82 l1) (road l82 l86) (road l83 l19) (road l83 l73) (road l84 l0) (road l84 l23) (road l85 l19) (road l85 l51) (road l86 l6) (road l86 l82) (road l87 l24) (road l87 l57) (road l88 l26) (road l88 l27) (road l89 l40) (road l89 l94) (road l9 l40) (road l9 l47) (road l90 l14) (road l90 l29) (road l91 l30) (road l91 l58) (road l92 l61) (road l92 l7) (road l92 l77) (road l93 l8) (road l93 l80) (road l94 l75) (road l94 l89) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l16) (spare-in l17) (spare-in l19) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l26) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l40) (spare-in l42) (spare-in l45) (spare-in l47) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l55) (spare-in l56) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l68) (spare-in l69) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l77) (spare-in l78) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l85) (spare-in l86) (spare-in l88) (spare-in l89) (spare-in l9) (spare-in l93) (spare-in l94) (vehicle-at l2))
    (:goal (vehicle-at l44))
)