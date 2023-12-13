(define (problem tireworld-092-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 l91 - location)
    (:init (road l0 l45) (road l0 l77) (road l1 l6) (road l1 l77) (road l10 l36) (road l10 l75) (road l11 l22) (road l11 l68) (road l12 l32) (road l12 l58) (road l13 l52) (road l13 l76) (road l14 l30) (road l14 l61) (road l15 l23) (road l15 l41) (road l15 l52) (road l16 l71) (road l16 l80) (road l17 l30) (road l17 l85) (road l18 l59) (road l18 l89) (road l19 l56) (road l19 l70) (road l2 l34) (road l2 l9) (road l20 l71) (road l20 l77) (road l21 l4) (road l21 l50) (road l22 l11) (road l22 l29) (road l23 l15) (road l23 l47) (road l24 l35) (road l24 l39) (road l25 l44) (road l25 l62) (road l25 l85) (road l26 l78) (road l26 l83) (road l27 l42) (road l27 l72) (road l28 l50) (road l28 l88) (road l29 l22) (road l29 l90) (road l3 l65) (road l3 l82) (road l30 l14) (road l30 l17) (road l31 l49) (road l31 l87) (road l32 l12) (road l32 l33) (road l32 l37) (road l33 l32) (road l33 l76) (road l34 l2) (road l34 l5) (road l35 l24) (road l35 l74) (road l36 l10) (road l36 l47) (road l36 l58) (road l37 l32) (road l37 l59) (road l38 l54) (road l38 l7) (road l39 l24) (road l39 l48) (road l4 l21) (road l4 l73) (road l40 l57) (road l40 l69) (road l41 l15) (road l41 l54) (road l42 l27) (road l42 l46) (road l43 l74) (road l44 l25) (road l44 l82) (road l45 l0) (road l45 l69) (road l45 l89) (road l46 l42) (road l46 l48) (road l47 l23) (road l47 l36) (road l48 l39) (road l48 l46) (road l49 l31) (road l49 l9) (road l5 l34) (road l5 l53) (road l50 l21) (road l50 l28) (road l51 l55) (road l51 l63) (road l51 l86) (road l52 l13) (road l52 l15) (road l53 l5) (road l53 l91) (road l54 l38) (road l54 l41) (road l55 l51) (road l55 l6) (road l56 l19) (road l56 l66) (road l57 l40) (road l57 l6) (road l58 l12) (road l58 l36) (road l58 l83) (road l59 l18) (road l59 l37) (road l6 l1) (road l6 l55) (road l6 l57) (road l6 l86) (road l60 l66) (road l60 l88) (road l61 l14) (road l61 l73) (road l62 l25) (road l62 l67) (road l63 l51) (road l63 l8) (road l64 l70) (road l64 l79) (road l65 l3) (road l66 l56) (road l66 l60) (road l67 l62) (road l67 l81) (road l68 l11) (road l68 l72) (road l69 l40) (road l69 l45) (road l7 l38) (road l7 l81) (road l70 l19) (road l70 l64) (road l71 l16) (road l71 l20) (road l72 l27) (road l72 l68) (road l73 l4) (road l73 l61) (road l74 l35) (road l74 l43) (road l75 l10) (road l75 l91) (road l76 l13) (road l76 l33) (road l77 l0) (road l77 l1) (road l77 l20) (road l78 l26) (road l78 l8) (road l79 l64) (road l79 l80) (road l8 l63) (road l8 l78) (road l80 l16) (road l80 l79) (road l81 l67) (road l81 l7) (road l82 l3) (road l82 l44) (road l83 l26) (road l83 l58) (road l84 l87) (road l84 l90) (road l85 l17) (road l85 l25) (road l86 l51) (road l86 l6) (road l87 l31) (road l87 l84) (road l88 l28) (road l88 l60) (road l89 l18) (road l89 l45) (road l9 l2) (road l9 l49) (road l90 l29) (road l90 l84) (road l91 l53) (road l91 l75) (spare-in l0) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l14) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l41) (spare-in l42) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l53) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l64) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l77) (spare-in l79) (spare-in l80) (spare-in l82) (spare-in l84) (spare-in l85) (spare-in l87) (spare-in l88) (spare-in l89) (spare-in l9) (spare-in l90) (spare-in l91) (vehicle-at l43))
    (:goal (vehicle-at l65))
)