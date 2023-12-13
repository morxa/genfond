(define (problem tireworld-090-06)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 - location)
    (:init (road l0 l5) (road l0 l50) (road l1 l13) (road l1 l24) (road l10 l27) (road l10 l47) (road l11 l33) (road l11 l44) (road l12 l4) (road l12 l45) (road l13 l1) (road l13 l74) (road l14 l48) (road l14 l54) (road l15 l31) (road l15 l64) (road l16 l60) (road l16 l82) (road l17 l59) (road l17 l72) (road l18 l22) (road l18 l30) (road l18 l57) (road l19 l49) (road l19 l85) (road l2 l38) (road l2 l8) (road l20 l43) (road l20 l88) (road l21 l33) (road l21 l74) (road l22 l18) (road l22 l67) (road l22 l87) (road l23 l27) (road l23 l52) (road l24 l1) (road l24 l65) (road l25 l59) (road l25 l67) (road l26 l35) (road l26 l83) (road l27 l10) (road l27 l23) (road l28 l68) (road l28 l79) (road l29 l42) (road l29 l64) (road l3 l6) (road l3 l73) (road l30 l18) (road l30 l54) (road l31 l15) (road l31 l34) (road l32 l55) (road l32 l66) (road l33 l11) (road l33 l21) (road l34 l31) (road l34 l56) (road l35 l26) (road l35 l40) (road l36 l37) (road l36 l75) (road l37 l36) (road l37 l68) (road l38 l2) (road l38 l86) (road l39 l46) (road l4 l12) (road l4 l65) (road l40 l35) (road l40 l56) (road l41 l48) (road l42 l29) (road l42 l85) (road l43 l20) (road l43 l69) (road l43 l77) (road l44 l11) (road l44 l78) (road l45 l12) (road l45 l79) (road l46 l39) (road l46 l70) (road l47 l10) (road l47 l83) (road l48 l14) (road l48 l41) (road l49 l19) (road l49 l82) (road l5 l0) (road l5 l88) (road l50 l0) (road l50 l73) (road l51 l58) (road l51 l63) (road l52 l23) (road l52 l76) (road l53 l80) (road l53 l87) (road l54 l14) (road l54 l30) (road l55 l32) (road l55 l58) (road l56 l34) (road l56 l40) (road l57 l18) (road l57 l89) (road l58 l51) (road l58 l55) (road l58 l71) (road l59 l17) (road l59 l25) (road l59 l8) (road l6 l3) (road l6 l80) (road l60 l16) (road l60 l86) (road l61 l7) (road l61 l75) (road l62 l66) (road l62 l84) (road l63 l51) (road l63 l70) (road l64 l15) (road l64 l29) (road l65 l24) (road l65 l4) (road l66 l32) (road l66 l62) (road l67 l22) (road l67 l25) (road l68 l28) (road l68 l37) (road l69 l43) (road l69 l71) (road l7 l61) (road l7 l84) (road l70 l46) (road l70 l63) (road l71 l58) (road l71 l69) (road l72 l17) (road l72 l77) (road l73 l3) (road l73 l50) (road l74 l13) (road l74 l21) (road l75 l36) (road l75 l61) (road l76 l52) (road l76 l9) (road l77 l43) (road l77 l72) (road l78 l44) (road l78 l81) (road l79 l28) (road l79 l45) (road l8 l2) (road l8 l59) (road l80 l53) (road l80 l6) (road l81 l78) (road l81 l9) (road l82 l16) (road l82 l49) (road l83 l26) (road l83 l47) (road l84 l62) (road l84 l7) (road l85 l19) (road l85 l42) (road l86 l38) (road l86 l60) (road l87 l22) (road l87 l53) (road l88 l20) (road l88 l5) (road l89 l57) (road l89 l9) (road l9 l76) (road l9 l81) (road l9 l89) (spare-in l1) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l18) (spare-in l21) (spare-in l24) (spare-in l28) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l36) (spare-in l37) (spare-in l4) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l48) (spare-in l51) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l58) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l65) (spare-in l66) (spare-in l68) (spare-in l7) (spare-in l70) (spare-in l74) (spare-in l75) (spare-in l78) (spare-in l79) (spare-in l81) (spare-in l84) (spare-in l89) (spare-in l9) (vehicle-at l39))
    (:goal (vehicle-at l41))
)